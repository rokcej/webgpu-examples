"use strict";

import { renderSource } from "./shaders.js";
import * as util from "../util.js";

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

// MSAA
const sampleCount = 4;

// Output textures
let depthTexture, depthTextureView;
let colorTexture, colorTextureView;
let msaaTexture, msaaTextureView;

function configureContext() {
    context.configure({
        device: device,
        format: preferredFormat,
        // size: [canvas.width, canvas.height, 1], // Deprecated
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        alphaMode: "opaque"
    });

    // if (depthTexture) {
    //     depthTexture.destroy();
    // }
    // depthTexture = device.createTexture({
    //     size: [canvas.width, canvas.height, 1],
    //     sampleCount: sampleCount,
    //     format: "depth24plus",
    //     usage: GPUTextureUsage.RENDER_ATTACHMENT
    // });
    // depthTextureView = depthTexture.createView();

    if (colorTexture) {
        colorTexture.destroy();
    }
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    if (msaaTexture) {
        msaaTexture.destroy();
    }
    msaaTexture = device.createTexture({
        size: [canvas.width, canvas.height, 1],
        sampleCount: sampleCount,
        format: preferredFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    msaaTextureView = msaaTexture.createView();
}
configureContext();

// Shaders
let renderModule = device.createShaderModule({ code: renderSource });

// Buffers
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
    size: 20,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(renderUniformBuffer, 0, new Float32Array([
    0.1, 0.3, 0.5,
    0.001
]));


// Pipeline
const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    // Vertex shader
    vertex: {
        module: renderModule,
        entryPoint: "vs_main",
        buffers: [ 
            { // Quad vertex buffer
                attributes: [
                    { // Position
                        shaderLocation: 0, // @location(0)
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
            format: preferredFormat,
            // constants: { todo, }
        }],
    },
    // Rasterization
    primitive: {
        frontFace: "ccw",
        cullMode: "back",
        topology: "triangle-list"
    },
    // Multisampling
    multisample: {
        count: sampleCount
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

// Draw
util.initFps();
render();
function render() {
    util.updateFps();

    // Swap framebuffer
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    const commandEncoder = device.createCommandEncoder();

    { // Render
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: msaaTextureView,
                resolveTarget: colorTextureView,
                clearValue: [0, 0, 0, 1],
                loadOp: "clear",
                storeOp: "store"
            }]
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setVertexBuffer(0, quadVertexBuffer);
        renderPass.setBindGroup(0, renderUniformBindGroup);
        renderPass.draw(6, 1, 0, 0);
        renderPass.end();
    }

    device.queue.submit([commandEncoder.finish()]);

    window.requestAnimationFrame(render);
};
