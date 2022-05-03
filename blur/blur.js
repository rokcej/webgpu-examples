/**
 * This program renders the final image in 3 stages:
 * 1. draw main scene, 2. apply Gaussian blur, 3. draw fullscreen quad
 * 
 * References:
 * - https://alain.xyz/blog/raw-webgpu
 * - https://github.com/tsherif/webgpu-examples
 * - https://austin-eng.com/webgpu-samples/
 */

"use strict";

import { mat4 } from "../lib//gl-matrix/esm/index.js";
import * as dat from "../lib/dat.gui/dat.gui.module.js"
import { coloredCube } from "../meshes.js";
import * as util from "../util.js";

import { GaussianBlurRender } from "./gaussian_blur_render.js";
import { GaussianBlurCompute } from "./gaussian_blur_compute.js";

// Check WebGPU support
if (!window.navigator.gpu) {
    document.body.innerHTML = "<div class=\"text\">Your browser does not support WebGPU</div>";
    throw new Error("WebGPU not supported");
}

// Gaussian blur settings
const settings = {
    numPasses: 20,
    useRenderPipeline: true,
    useComputePipeline: false
}
// dat.gui
const gui = new dat.GUI();
gui.width = 320;
(() => { // Don't pollute the namespace
    const folder = gui.addFolder("Gaussian Blur");
    folder.add(settings, "numPasses", 0, 100).step(1).name("Number of Passes");
    const renderToggle = folder.add(settings, "useRenderPipeline").name("Use Render Pipeline");
    const computeToggle = folder.add(settings, "useComputePipeline").name("Use Compute Pipeline");
    const updateToggle = (useRender) => {
        settings.useRenderPipeline = useRender;
        settings.useComputePipeline = !useRender;
        renderToggle.updateDisplay();
        computeToggle.updateDisplay();
    };
    renderToggle.onChange(useRender => updateToggle(useRender));
    computeToggle.onChange(useCompute => updateToggle(!useCompute));
    folder.open();
})();

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
    usage: GPUTextureUsage.RENDER_ATTACHMENT, // GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_SRC
    compositingAlphaMode: "opaque"
});

// Textures
const sampler = device.createSampler({
    minFilter: "nearest",
    magFilter: "nearest"
});
/// Main scene
let sceneTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT
});
let sceneTextureView = sceneTexture.createView();
let sceneDepthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus-stencil8",
    usage: GPUTextureUsage.RENDER_ATTACHMENT // | GPUTextureUsage.COPY_SRC
});
let sceneDepthTextureView = sceneDepthTexture.createView();
/// Output
let colorTexture = context.getCurrentTexture();
let colorTextureView = colorTexture.createView();

// Buffers
let vertexBuffer = createBuffer(device, coloredCube.vertices, GPUBufferUsage.VERTEX);
let indexBuffer = createBuffer(device, coloredCube.indices, GPUBufferUsage.INDEX);

// Shaders
const vsQuadSource = `
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragUV: vec2<f32>
};

@stage(vertex)
fn main(@builtin(vertex_index) VertexIndex : u32) -> VSOut {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0), vec2<f32>(-1.0, -1.0)
    );
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0)
    );
    var vsOut: VSOut;
    vsOut.position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
    vsOut.fragUV = uv[VertexIndex];
    return vsOut;
}
`;
const fsQuadSource = `
@group(0) @binding(0) var uSampler: sampler;
@group(0) @binding(1) var uTexture: texture_2d<f32>;

@stage(fragment)
fn main(@location(0) fragUV: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(uTexture, uSampler, fragUV);
}
`;
const vsSceneSource = `
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>
};

struct UBO {
    mvpMat: mat4x4<f32>
};
@binding(0) @group(0) var<uniform> uniforms: UBO;

@stage(vertex)
fn main(@location(0) inPos: vec3<f32>,
        @location(1) inColor: vec3<f32>) -> VSOut {
    var vsOut: VSOut;
    vsOut.position = uniforms.mvpMat * vec4<f32>(inPos, 1.0);
    vsOut.color = inColor;
    return vsOut;
}
`;
const fsSceneSource = `
@stage(fragment)
fn main(@location(0) inColor: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(inColor, 1.0);
}
`;
let vsQuadModule = device.createShaderModule({ code: vsQuadSource });
let fsQuadModule = device.createShaderModule({ code: fsQuadSource });
let vsSceneModule = device.createShaderModule({ code: vsSceneSource });
let fsSceneModule = device.createShaderModule({ code: fsSceneSource });

// Uniform data
const projMat = mat4.create();
const viewMat = mat4.create();
const pvMat = mat4.create();

mat4.perspective(projMat, Math.PI / 2, canvas.width / canvas.height, 0.1, 100.0);
mat4.lookAt(viewMat, [0, 0, 2], [0, 0, 0], [0, 1, 0]);
mat4.mul(pvMat, projMat, viewMat);

// Uniforms
let sceneBuffer = createBuffer(device, pvMat, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

// Gaussian blur pipelines
const gaussianBlurRender = new GaussianBlurRender(device, [canvas.width, canvas.height, 1], sceneTexture, sceneTextureView, vsQuadModule);
const gaussianBlurCompute = new GaussianBlurCompute(device, [canvas.width, canvas.height, 1], sceneTexture, sceneTextureView);

// Uniform bind group
let quadBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: "filtering" }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: "float" }
        }
    ]
});
let quadBindGroupRender = device.createBindGroup({
    layout: quadBindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: sampler
        },
        {
            binding: 1,
            resource: gaussianBlurRender.getOutputTextureView()
        }
    ]
});
let quadBindGroupCompute = device.createBindGroup({
    layout: quadBindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: sampler
        },
        {
            binding: 1,
            resource: gaussianBlurCompute.getOutputTextureView()
        }
    ]
});
let sceneBindGroupLayout = device.createBindGroupLayout({
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" }
    }]
});
let sceneBindGroup = device.createBindGroup({
    layout: sceneBindGroupLayout,
    entries: [{
        binding: 0,
        resource: { buffer: sceneBuffer }
    }]
});

// Render pipeline
const quadPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
        bindGroupLayouts: [quadBindGroupLayout]
    }),
    // Vertex shader
    vertex: {
        module: vsQuadModule,
        entryPoint: "main",
    },
    // Fragment shader
    fragment: {
        module: fsQuadModule,
        entryPoint: "main",
        targets: [{ format: preferredFormat }],
    },
    // Rasterization
    primitive: {
        // frontFace: "ccw",
        // cullMode: "back",
        topology: "triangle-list"
    }
});
const scenePipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
        bindGroupLayouts: [sceneBindGroupLayout]
    }),
    // Vertex shader
    vertex: {
        module: vsSceneModule,
        entryPoint: "main",
        buffers: [
            { // vertexBuffer
                attributes: [
                    { // Position
                        shaderLocation: 0, // @location(0)
                        offset: 0,
                        format: "float32x3"
                    },
                    { // Color
                        shaderLocation: 1, // @location(1)
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
        module: fsSceneModule,
        entryPoint: "main",
        targets: [{
            format: "rgba8unorm"
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
util.initFps();
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

function render() {
    util.updateFps();
    // Update MVP matrix
    let t = performance.now() / 1000;

    const modelMat = mat4.create();
    mat4.rotateX(modelMat, modelMat, t * 0.75);
    mat4.rotateY(modelMat, modelMat, t * 1.5);
    const pvmMat = mat4.create();
    mat4.mul(pvmMat, pvMat, modelMat);

    device.queue.writeBuffer(sceneBuffer, 0, pvmMat);

    // Swap framebuffer
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    const commandEncoder = device.createCommandEncoder();

    // Main scene
    const scenePass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: sceneTextureView,
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store"
        }],
        depthStencilAttachment: {
            view: sceneDepthTextureView,
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            stencilClearValue: 0,
            stencilLoadOp: "clear",
            stencilStoreOp: "store"
        }
    });
    scenePass.setPipeline(scenePipeline);
    scenePass.setVertexBuffer(0, vertexBuffer);
    scenePass.setIndexBuffer(indexBuffer, "uint16");
    scenePass.setBindGroup(0, sceneBindGroup);
    scenePass.drawIndexed(36);
    scenePass.end();

    // Gaussian blur
    const _useComputePipeline = settings.useComputePipeline; // We don't want any async shenanigans
    if (_useComputePipeline) {
        gaussianBlurCompute.render(commandEncoder, settings.numPasses);
    } else {
        gaussianBlurRender.render(commandEncoder, settings.numPasses);
    }

    // Display
    const quadPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: colorTextureView,
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store"
        }]
    });
    quadPass.setPipeline(quadPipeline);
    quadPass.setBindGroup(0, _useComputePipeline ? quadBindGroupCompute : quadBindGroupRender);
    quadPass.draw(6, 1, 0, 0);
    quadPass.end();

    device.queue.submit([commandEncoder.finish()]);

    window.requestAnimationFrame(render);
};
