/**
 * References:
 * - https://alain.xyz/blog/raw-webgpu
 * - https://github.com/tsherif/webgpu-examples
 */

"use strict";

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
    [[builtin(position)]] Position: vec4<f32>;
    [[location(0)]] color: vec3<f32>;
};

[[stage(vertex)]]
fn main([[location(0)]] inPos: vec3<f32>,
        [[location(1)]] inColor: vec3<f32>) -> VSOut {
    var vsOut: VSOut;
    vsOut.Position = vec4<f32>(inPos, 1.0);
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
    renderPass.setVertexBuffer(0, positionBuffer);
    renderPass.setVertexBuffer(1, colorBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint16");

    renderPass.drawIndexed(3);
    renderPass.endPass();

    device.queue.submit([commandEncoder.finish()]);
}

function render() {
    // Swap framebuffer
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    encodeCommands();

    window.requestAnimationFrame(render);
};
