"use strict";

import { mat4 } from "../lib/gl-matrix/esm/index.js";
import * as dat from "../lib/dat.gui/dat.gui.module.js"
import { renderSource, computeSource } from "./shaders.js";
import * as util from "../util.js";

// Check WebGPU support
if (!window.navigator.gpu) {
    document.body.innerHTML = "<div class=\"text\">Your browser does not support WebGPU</div>";
    throw new Error("WebGPU not supported");
}

// Particles
const urlParams = new URLSearchParams(window.location.search);
const numParticles = parseInt(urlParams.get("numParticles") || 1000000); // Default to 1 million
document.title = `WebGPU ${numParticles.toLocaleString()} Particles`

// MSAA
const sampleCount = 4;

// Curl noise settings
const settings = {
    flowScale: 15,
    flowEvolution: 0.1,
    flowSpeed: 0.8,
    artisticMode: false,
    mouseSensitivity: 2,
}
// dat.gui
const gui = new dat.GUI();
(() => { // Don't pollute the namespace
    const noiseFolder = gui.addFolder("Curl noise");
    noiseFolder.add(settings, "flowScale", 1, 30).step(0.1).name("Flow scale");
    noiseFolder.add(settings, "flowEvolution", 0.01, 1.0).step(0.01).name("Flow evolution");
    noiseFolder.add(settings, "flowSpeed", 0.01, 5).step(0.01).name("Flow speed");
    noiseFolder.add(settings, "artisticMode").name("Artistic mode");
    noiseFolder.open();
    const movementFolder = gui.addFolder("Movement");
    movementFolder.add(settings, "mouseSensitivity", 0.1, 10).step(0.1);
})();

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
    format: presentationFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT
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

// Buffers
const particleByteSize = (3 + 1 + 4 + 1 + 1 + 2) * 4; // Last 2 floats are padding
const particlePositionOffset = 0;
const particleColorOffset = 4 * 4;
let particlesBuffer = device.createBuffer({
    size: numParticles * particleByteSize,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE
});

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
    size: 8 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(computeUniformBuffer, 28, new Uint32Array([numParticles]))

// Uniform data
const projMat = mat4.create();
const viewMat = mat4.create();
const pvMat = mat4.create();
const camera = {
    pos: new Float32Array([0, 0, 0]),
    fov: Math.PI / 2,
    rot: [0, 0],
    near: 0.1,
    far: 100.0
}
mat4.perspective(projMat, camera.fov, canvas.width / canvas.height, camera.near, camera.far);
mat4.lookAt(viewMat, camera.pos, [0, 0, -1], [0, 1, 0]);
mat4.mul(pvMat, projMat, viewMat);

// Event handling
const mouse = {
    dragging: false,
    xPrev: 0,
    yPrev: 0
};
canvas.addEventListener("mousedown", (event) => {
    mouse.xPrev = event.clientX;
    mouse.yPrev = event.clientY;
    mouse.dragging = true;
});
canvas.addEventListener("mouseup", (event) => {
    mouse.dragging = false;
});
canvas.addEventListener("mousemove", (event) => {
    if (!(event.buttons & 1)) {
        mouse.dragging = false;
    } else if (mouse.dragging) {
        let x = event.clientX, y = event.clientY;
        let dx = x - mouse.xPrev;
        let dy = y - mouse.yPrev;
        mouse.xPrev = x;
        mouse.yPrev = y;

        camera.rot[0] -= dy * settings.mouseSensitivity * 0.001; // Pitch
        camera.rot[1] += dx * settings.mouseSensitivity * 0.001; // Yaw

        // Constrain horizontal rotation to [0, 2*PI)
        camera.rot[1] = (camera.rot[1] + 2.0 * Math.PI) % (2.0 * Math.PI);
        // Constrain vertical rotation to (-PI/2, PI/2)
        const eps = 0.001; // Small offset to prevent lookAt errors
        if (camera.rot[0] > 0.5 * Math.PI - eps)
            camera.rot[0] = 0.5 * Math.PI - eps
        else if (camera.rot[0] < -0.5 * Math.PI + eps)
            camera.rot[0] = -0.5 * Math.PI + eps;
    }
});


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
    mat4.lookAt(viewMat, [0, 0, 0], [
            Math.cos(camera.rot[1]) * Math.cos(camera.rot[0]),
            Math.sin(camera.rot[0]),
            Math.sin(camera.rot[1]) * Math.cos(camera.rot[0])
        ], [0, 1, 0]
    );
    mat4.multiply(pvMat, projMat, viewMat);
    device.queue.writeBuffer(renderUniformBuffer, 0, pvMat);
    device.queue.writeBuffer(renderUniformBuffer, 64, new Float32Array([
        viewMat[0], viewMat[4], viewMat[8], 0, // vec3 is 16-byte aligned
        viewMat[1], viewMat[5], viewMat[9], 0
    ]));
    device.queue.writeBuffer(computeUniformBuffer, 0, new Float32Array([
        Math.random(), dt, t,
        settings.flowScale, settings.flowEvolution, settings.flowSpeed
    ]));
    device.queue.writeBuffer(computeUniformBuffer, 24, new Uint32Array([settings.artisticMode]));

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
