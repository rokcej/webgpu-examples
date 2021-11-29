/**
 * References:
 * - https://alain.xyz/blog/raw-webgpu
 * - https://github.com/tsherif/webgpu-examples
 */

"use strict";

import { mat4 } from "../lib/gl-matrix-esm/index.js";
import { coloredCube } from "../meshes.js";

// Check WebGPU support
if (!window.navigator.gpu) {
    document.body.innerHTML = "<div class=\"text\">Your browser does not support WebGPU</div>";
    throw new Error("WebGPU not supported");
}

// Device
const adapter = await window.navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Context
const canvas = document.getElementById("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const context = canvas.getContext("webgpu");
const preferredFormat = context.getPreferredFormat(adapter);
context.configure({
    device: device,
    format: preferredFormat, // "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT // GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_SRC
});

// Textures
let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    mipLevelCount: 1, // Optional
    sampleCount: 1, // Optional
    dimension: "2d", // Optional
    format: "depth24plus-stencil8",
    usage: GPUTextureUsage.RENDER_ATTACHMENT // | GPUTextureUsage.COPY_SRC
});
let depthTextureView = depthTexture.createView();
let colorTexture = context.getCurrentTexture();
let colorTextureView = colorTexture.createView();

// Buffers
let vertexBuffer = createBuffer(device, coloredCube.vertices, GPUBufferUsage.VERTEX);
let indexBuffer = createBuffer(device, coloredCube.indices, GPUBufferUsage.INDEX);

// Shaders
const vsSource = `
struct VSOut {
    [[builtin(position)]] Position: vec4<f32>;
    [[location(0)]] color: vec3<f32>;
};

[[block]] struct UBO {
    mvpMat: mat4x4<f32>;
};
[[binding(0), group(0)]] var<uniform> uniforms: UBO;

[[stage(vertex)]]
fn main([[location(0)]] inPos: vec3<f32>,
        [[location(1)]] inColor: vec3<f32>) -> VSOut {
    var vsOut: VSOut;
    vsOut.Position = uniforms.mvpMat * vec4<f32>(inPos, 1.0);
    vsOut.color = inColor;
    return vsOut;
}
`;
const fsSource = `
[[stage(fragment)]]
fn main([[location(0)]] inColor: vec3<f32>) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(inColor, 1.0);
}
`;
let vsModule = device.createShaderModule({ code: vsSource });
let fsModule = device.createShaderModule({ code: fsSource });

// Uniform data
const projMat = mat4.create();
const viewMat = mat4.create();
const pvMat = mat4.create();
const pvmMat = mat4.create();

mat4.perspective(projMat, Math.PI / 2, canvas.width / canvas.height, 0.1, 100.0);
mat4.lookAt(viewMat, [0, 0, 2], [0, 0, 0], [0, 1, 0]);
mat4.mul(pvMat, projMat, viewMat);

// Uniforms
let uniformBuffer = createBuffer(device, pvMat, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

// Uniform bind group
let uniformBindGroupLayout = device.createBindGroupLayout({
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
            type: "uniform"
        }
    }]
});
let uniformBindGroup = device.createBindGroup({
    layout: uniformBindGroupLayout,
    entries: [{
        binding: 0,
        resource: {
            buffer: uniformBuffer
        }
    }]
});

// Render pipeline
let pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [uniformBindGroupLayout]
});
const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    // Vertex shader
    vertex: {
        module: vsModule,
        entryPoint: "main",
        buffers: [
            { // vertexBuffer
                attributes: [
                    { // Position
                        shaderLocation: 0, // [[location(0)]]
                        offset: 0,
                        format: "float32x3"
                    },
                    { // Color
                        shaderLocation: 1, // [[location(1)]]
                        offset: 4 * 3,
                        format: "float32x3"
                    }
                ],
                arrayStride: 4 * 6, // sizeof(float) * 6
                stepMode: "vertex"
            }
        ]
    },
    // Fragment shader
    fragment: {
        module: fsModule,
        entryPoint: "main",
        targets: [{
            format: preferredFormat
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
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus-stencil8"
    }
});

// Draw
render();



function createBuffer(device, arr, usage) {
    let buffer = device.createBuffer({
        size: ((arr.byteLength + 3) & ~3), // Is alignment necessary?
        usage: usage,
        mappedAtCreation: true
    });

    const writeArray = arr instanceof Uint16Array ?
        new Uint16Array(buffer.getMappedRange()) : new Float32Array(buffer.getMappedRange());
    writeArray.set(arr);
    buffer.unmap();
    return buffer;
};

function encodeCommands() {
    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: colorTextureView,
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

    renderPass.setPipeline(pipeline);

    renderPass.setViewport(0, 0, canvas.width, canvas.height, 0, 1);
    renderPass.setScissorRect(0, 0, canvas.width, canvas.height);

    // Attributes
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint16");

    // Uniforms
    renderPass.setBindGroup(0, uniformBindGroup);

    renderPass.drawIndexed(36);
    renderPass.endPass();

    device.queue.submit([commandEncoder.finish()]);
}

function render() {
    // Update MVP matrix
    let t = performance.now() / 1000;

    const modelMat = mat4.create();
    mat4.rotateX(modelMat, modelMat, t * 0.75);
    mat4.rotateY(modelMat, modelMat, t * 1.5);

    mat4.mul(pvmMat, pvMat, modelMat);

    device.queue.writeBuffer(uniformBuffer, 0, pvmMat);

    // Swap framebuffer
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    encodeCommands();

    window.requestAnimationFrame(render);
};
