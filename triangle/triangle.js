/**
 * References:
 * - https://alain.xyz/blog/raw-webgpu
 * - https://github.com/tsherif/webgpu-examples
 */

"use strict";

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
const preferredFormat = navigator.gpu.getPreferredCanvasFormat()
context.configure({
    device: device,
    format: preferredFormat, // "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT, // GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    alphaMode: "opaque"
});

// Textures
const sampleCount = 4;
let colorTexture = context.getCurrentTexture();
let colorTextureView = colorTexture.createView();
let msaaTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    sampleCount: sampleCount,
    format: preferredFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT
});
let msaaTextureView = msaaTexture.createView();

// Data
const positions = new Float32Array([
    0.5, -0.5, 0.0,
   -0.5, -0.5, 0.0,
    0.0,  0.5, 0.0
]);
const colors = new Float32Array([
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
]);
const indices = new Uint16Array([ 0, 1, 2 ]);

// Buffers
let positionBuffer = createBuffer(device, positions, GPUBufferUsage.VERTEX);
let colorBuffer = createBuffer(device, colors, GPUBufferUsage.VERTEX);
let indexBuffer = createBuffer(device, indices, GPUBufferUsage.INDEX);

// Shaders
const vsSource = `
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>
};

@vertex
fn main(@location(0) inPos: vec3<f32>,
        @location(1) inColor: vec3<f32>) -> VSOut {
    var vsOut: VSOut;
    vsOut.position = vec4<f32>(inPos, 1.0);
    vsOut.color = inColor;
    return vsOut;
}
`;
const fsSource = `
@fragment
fn main(@location(0) inColor: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(inColor, 1.0);
}
`;
let vsModule = device.createShaderModule({ code: vsSource });
let fsModule = device.createShaderModule({ code: fsSource });

// Render pipeline
let pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: []
});
const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    // Vertex shader
    vertex: {
        module: vsModule,
        entryPoint: "main",
        buffers: [
            { // Position
                attributes: [{
                    shaderLocation: 0, // [[location(0)]]
                    offset: 0,
                    format: "float32x3"
                }],
                arrayStride: 4 * 3, // sizeof(float) * 3
                stepMode: "vertex"
            },
            { // Color
                attributes: [{
                    shaderLocation: 1, // [[location(1)]]
                    offset: 0,
                    format: "float32x3"
                }],
                arrayStride: 4 * 3, // sizeof(float) * 3
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
        frontFace: "cw",
        cullMode: "none",
        topology: "triangle-list"
    },
    // Multisampling
    multisample: {
        count: sampleCount
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
            view: msaaTextureView,
            resolveTarget: colorTextureView,
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store"
        }]
    });

    renderPass.setPipeline(pipeline);

    renderPass.setViewport(0, 0, canvas.width, canvas.height, 0, 1);
    renderPass.setScissorRect(0, 0, canvas.width, canvas.height);

    // Attributes
    renderPass.setVertexBuffer(0, positionBuffer);
    renderPass.setVertexBuffer(1, colorBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint16");

    renderPass.drawIndexed(3);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
}

function render() {
    // Swap framebuffer
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    encodeCommands();

    window.requestAnimationFrame(render);
};
