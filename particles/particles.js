"use strict";

import { mat4 } from "../lib/gl-matrix/esm/index.js";
import { renderSource, computeSource } from "./shaders.js";
import * as util from "../util.js";

// Check WebGPU support
if (!window.navigator.gpu) {
    document.body.innerHTML = "<div class=\"text\">Your browser does not support WebGPU</div>";
    throw new Error("WebGPU not supported");
}

const numParticles = 1000000;
const particleByteSize = (3 + 1 + 4 + 3 + 1) * 4;
const particlePositionOffset = 0;
const particleColorOffset = 4 * 4;

const sampleCount = 4; // MSAA

// Device
const adapter = await window.navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Context
const canvas = document.getElementById("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const context = canvas.getContext("webgpu");
const presentationFormat = context.getPreferredFormat(adapter);
context.configure({
    device: device,
    format: presentationFormat, // "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT // GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_SRC
});

// Output textures
let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    sampleCount: sampleCount,
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT
});
let depthTextureView = depthTexture.createView();
let colorTexture = context.getCurrentTexture();
let colorTextureView = colorTexture.createView();
let msaaTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    sampleCount: sampleCount,
    format: presentationFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT
});
let msaaTextureView = msaaTexture.createView();

// Shaders
let renderModule = device.createShaderModule({ code: renderSource });
let computeModule = device.createShaderModule({ code: computeSource });

// TEMPORARY
// TODO: Remove this
let sz = particleByteSize / 4;
let tmp = new Float32Array(numParticles * sz);
let dim = Math.pow(numParticles, 1/3);
for (let i = 0; i < numParticles; ++i) {
    let i_copy = i;
    let x = i_copy % dim;
    i_copy = Math.floor(i_copy/dim);
    let y = i_copy % dim;
    let z = Math.floor(i_copy/dim);

    tmp[i * sz + 0] = x-dim/2;
    tmp[i * sz + 1] = y-dim/2;
    tmp[i * sz + 2] = -z - 1;
    tmp[i * sz + 4] = 1;
    tmp[i * sz + 5] = 1;
    tmp[i * sz + 6] = 1;
    tmp[i * sz + 7] = 1;
    tmp[i * sz + 8] = 0;
    tmp[i * sz + 9] = 10;
    tmp[i * sz + 10] = 0;
}

// Buffers
let particlesBuffer = device.createBuffer({
    size: numParticles * particleByteSize,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST // TODO: Remove COPY_DST
});
device.queue.writeBuffer(particlesBuffer, 0, tmp);

let quadVertexBuffer = device.createBuffer({
    size: 6 * 2 * 4,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true
});
new Float32Array(quadVertexBuffer.getMappedRange()).set(new Float32Array([
    -1.0, -1.0, +1.0, -1.0, +1.0, +1.0, +1.0, +1.0, -1.0, +1.0, -1.0, -1.0
]));
quadVertexBuffer.unmap();

let renderUniformBuffer = device.createBuffer({
    size: (4 * 4 + 4 + 4) * 4, // vec3 is 16-byte aligned
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

let computeUniformBuffer = device.createBuffer({
    size: (4 + 1 + 1) * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(computeUniformBuffer, 20, new Uint32Array([numParticles]))

// Uniform data
const projMat = mat4.create();
const viewMat = mat4.create();
const pvMat = mat4.create();
const cameraPos = new Float32Array([0, 0, 2]);
mat4.perspective(projMat, Math.PI / 2, canvas.width / canvas.height, 0.1, 100.0);
mat4.lookAt(viewMat, cameraPos, [0, 0, 0], [0, 1, 0]);
mat4.mul(pvMat, projMat, viewMat);

// Pipeline
const renderPipeline = device.createRenderPipeline({
    // Vertex shader
    vertex: {
        module: renderModule,
        entryPoint: "vs_main",
        buffers: [
            { // Particle instance buffer
                attributes: [
                    { // Position
                        shaderLocation: 0, // [[location(0)]]
                        offset: particlePositionOffset,
                        format: "float32x3"
                    },
                    { // Color
                        shaderLocation: 1, // [[location(1)]]
                        offset: particleColorOffset,
                        format: "float32x4"
                    },
                ],
                arrayStride: particleByteSize,
                stepMode: "instance"
            }, 
            { // Quad vertex buffer
                attributes: [
                    { // Position
                        shaderLocation: 2, // [[location(2)]]
                        offset: 0,
                        format: "float32x2"
                    }
                ],
                arrayStride: 2 * 4,
                stepMode: "vertex"
            }
        ]
    },
    // Fragment shader
    fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{
            format: presentationFormat,
            blend: {
                color: {
                    operation: "add",
                    srcFactor: "src-alpha",
                    dstFactor: "one"
                },
                alpha: {
                    operation: "add",
                    srcFactor: "zero",
                    dstFactor: "one"
                }
            }
        }],
    },
    // Rasterization
    primitive: {
        frontFace: "ccw",
        cullMode: "back",
        topology: "triangle-list"
    },
    // Depth test
    depthStencil: {
        depthWriteEnabled: false,
        depthCompare: "less",
        format: "depth24plus"
    },
    // Multisampling
    multisample: {
        count: sampleCount
    }
});
const computePipeline = device.createComputePipeline({
    compute: {
        module: computeModule,
        entryPoint: "main"
    }
});

// Uniform bind group
const renderUniformBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: {
                buffer: renderUniformBuffer
            }
        }
    ]
});
// Note: if you let WebGPU create bind group layout implicitly,
// it will omit unused bindings
const computeUniformBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0), 
    entries: [
        {
            binding: 0,
            resource: { buffer: computeUniformBuffer }
        },
        {
            binding: 1,
            resource: { buffer: particlesBuffer }
        }
    ]
});

// Draw
let tPrev = performance.now() * 0.001;
util.initFps();
render();


function render() {
    util.updateFps();
    // Time
    let t = performance.now() * 0.001;
    let dt = t - tPrev;
    tPrev = t;

    // Update uniforms
    device.queue.writeBuffer(renderUniformBuffer, 0, pvMat);
    device.queue.writeBuffer(renderUniformBuffer, 64, new Float32Array([
        viewMat[0], viewMat[4], viewMat[8], 0, // vec3 is 16-byte aligned
        viewMat[1], viewMat[5], viewMat[9], 0
    ]));
    device.queue.writeBuffer(computeUniformBuffer, 0, new Float32Array([
        0.0, 0.0, 0.0, 0.0, dt
    ]));

    // Swap framebuffer
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    const commandEncoder = device.createCommandEncoder();

    { // Compute
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeUniformBindGroup);
        computePass.dispatch(Math.ceil(numParticles / 256));
        computePass.endPass();
    }
    { // Render
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: msaaTextureView,
                resolveTarget: colorTextureView,
                loadValue: [0, 0, 0, 1],
                storeOp: "store"
            }],
            depthStencilAttachment: {
                view: depthTextureView,
                depthLoadValue: 1,
                depthStoreOp: "store",
                stencilLoadValue: 0,
                stencilStoreOp: "store"
            }
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setVertexBuffer(0, particlesBuffer);
        renderPass.setVertexBuffer(1, quadVertexBuffer);
        renderPass.setBindGroup(0, renderUniformBindGroup);
        renderPass.draw(6, numParticles, 0, 0);
        renderPass.endPass();
    }

    device.queue.submit([commandEncoder.finish()]);

    window.requestAnimationFrame(render);
};
