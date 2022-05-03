/**
 * References:
 * - https://alain.xyz/blog/raw-webgpu
 * - https://github.com/tsherif/webgpu-examples
 * - https://austin-eng.com/webgpu-samples/
 */

"use strict";

import { mat3, mat4 } from "../lib/gl-matrix/esm/index.js";
import { texturedCube } from "../meshes.js";

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
    usage: GPUTextureUsage.RENDER_ATTACHMENT, // GPUTextureUsage.OUTPUT_ATTACHMENT | GPUTextureUsage.COPY_SRC
    compositingAlphaMode: "opaque"
});

// Input textures
const sampler = device.createSampler({
    minFilter: "nearest",
    magFilter: "nearest"
});
const [texture, textureSpecular] = await Promise.all([
    loadTexture("../data/iron.png"),
    loadTexture("../data/specular.png")
]);
const textureView = texture.createView();
const textureSpecularView = textureSpecular.createView();

// Output textures
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
let vertexBuffer = createBuffer(device, texturedCube.vertices, GPUBufferUsage.VERTEX);
let indexBuffer = createBuffer(device, texturedCube.indices, GPUBufferUsage.INDEX);

// Shaders
const vsSource = `
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragPos: vec3<f32>,
    @location(1) fragUV: vec2<f32>,
    @location(2) fragNorm: vec3<f32>
};

struct UBO {
    modelMat: mat4x4<f32>,
    pvMat: mat4x4<f32>,
    normMat: mat3x3<f32> // Watch out for alignment
};
@group(0) @binding(0) var<uniform> ubo: UBO;

@stage(vertex)
fn main(@location(0) inPos: vec3<f32>,
        @location(2) inUV: vec2<f32>,
        @location(1) inNorm: vec3<f32>) -> VSOut {            
    var worldPos: vec4<f32> = ubo.modelMat * vec4<f32>(inPos, 1.0);

    var vsOut: VSOut;
    vsOut.position =  ubo.pvMat * worldPos;
    vsOut.fragPos = worldPos.xyz / worldPos.w;
    vsOut.fragUV = inUV;
    vsOut.fragNorm = ubo.normMat * inNorm;
    return vsOut;
}
`;
const fsSource = `
let shininess: f32 = 32.0;
let ambient: f32 = 0.1;

struct UBO {
    cameraPos: vec3<f32>,
    lightPos: vec3<f32>
};
@group(0) @binding(1) var<uniform> ubo: UBO;
@group(0) @binding(2) var uSampler: sampler;
@group(0) @binding(3) var uTexture: texture_2d<f32>;
@group(0) @binding(4) var uTextureSpecular: texture_2d<f32>;

@stage(fragment)
fn main(@location(0) fragPos: vec3<f32>,
        @location(1) fragUV: vec2<f32>,
        @location(2) fragNorm: vec3<f32>) -> @location(0) vec4<f32> {

    var V: vec3<f32> = normalize(ubo.cameraPos - fragPos);
    var L: vec3<f32> = normalize(ubo.lightPos - fragPos);
    var H: vec3<f32> = normalize(V + L);

    var specular: f32 = pow(max(dot(fragNorm, H), 0.0), shininess);
    var diffuse: f32 = max(dot(fragNorm, L), 0.0);
    var color: vec3<f32> = textureSample(uTexture, uSampler, fragUV).rgb;
    var specularStrength: f32 = textureSample(uTextureSpecular, uSampler, fragUV).r;

    return vec4<f32>(color * (ambient + diffuse + specularStrength * specular * 0.8), 1.0);
}
`;
let vsModule = device.createShaderModule({ code: vsSource });
let fsModule = device.createShaderModule({ code: fsSource });

// Uniform data
const projMat = mat4.create();
const viewMat = mat4.create();
const pvMat = mat4.create();

const cameraPos = new Float32Array([0, 0, 2]);
const lightPos = new Float32Array([0.2, 0.8, 2]);

mat4.perspective(projMat, Math.PI / 2, canvas.width / canvas.height, 0.1, 100.0);
mat4.lookAt(viewMat, cameraPos, [0, 0, 0], [0, 1, 0]);
mat4.mul(pvMat, projMat, viewMat);

// Uniforms
let vsUniformBuffer = device.createBuffer({
    size: 192, // 3 * [4x4] * sizeof(float) = 192
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(vsUniformBuffer, 64, pvMat);
let fsUniformBuffer = device.createBuffer({
    size: 32, // 2 * [3] * sizeof(float) = 24
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(fsUniformBuffer, 0, cameraPos);
device.queue.writeBuffer(fsUniformBuffer, 16, lightPos);

// Uniform bind group
let uniformBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: {
                type: "uniform"
            }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: {
                type: "uniform"
            }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: {
                type: "filtering"
            }
        },
        {
            binding: 3,
            visibility: GPUShaderStage.FRAGMENT,
            texture: {
                sampleType: "float"
            }
        },
        {
            binding: 4,
            visibility: GPUShaderStage.FRAGMENT,
            texture: {
                sampleType: "float"
            }
        }
    ]
});
let uniformBindGroup = device.createBindGroup({
    layout: uniformBindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: vsUniformBuffer
            }
        },
        {
            binding: 1,
            resource: {
                buffer: fsUniformBuffer
            }
        },
        {
            binding: 2,
            resource: sampler
        },
        {
            binding: 3,
            resource: textureView
        },
        {
            binding: 4,
            resource: textureSpecularView
        }
    ]
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
                    { // UV
                        shaderLocation: 2, // [[location(2)]]
                        offset: 4 * 3, // sizeof(float) * 3
                        format: "float32x2"
                    },
                    { // Normal
                        shaderLocation: 1, // [[location(1)]]
                        offset: 4 * 5, // sizeof(float) * 5
                        format: "float32x3"
                    }
                ],
                arrayStride: 4 * 8, // sizeof(float) * 6
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

async function loadTexture(url) {
    const textureCanvas = document.createElement("canvas");
    const textureCanvasCtx = textureCanvas.getContext("2d");

    const img = document.createElement("img");
    img.src = url;
    await img.decode();
    textureCanvas.width = img.width;
    textureCanvas.height = img.height;
    textureCanvasCtx.drawImage(img, 0, 0);
    const imgData = textureCanvasCtx.getImageData(0, 0, img.width, img.height).data;

    const texture = device.createTexture({
        size: [img.width, img.height, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
    });
    device.queue.writeTexture(
        {texture: texture},
        imgData,
        {
            offset: 0,
            bytesPerRow: img.width * 4, // 4 8-bit channels
            rowsPerImage: img.height
        },
        [img.width, img.height, 1]
    );
    return texture;
}

function encodeCommands() {
    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: colorTextureView,
            clearValue: [0, 0, 0, 1],
            loadOp: "clear",
            storeOp: "store"
        }],
        depthStencilAttachment: {
            view: depthTextureView,
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            stencilClearValue: 0,
            stencilLoadOp: "clear",
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
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
}

function render() {
    // Update MVP matrix
    let t = performance.now() / 1000;

    const modelMat = mat4.create();
    mat4.rotateX(modelMat, modelMat, t * 0.75);
    mat4.rotateY(modelMat, modelMat, t * 1.5);

    // normMat = transpose(inverse(mat3(modelMat)))
    // http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/the-normal-matrix/
    const normMat = mat4.create();
    mat4.invert(normMat, modelMat);
    mat4.transpose(normMat, normMat);

    // I used a 4x4 normal matrix because 3x3 matrices in WebGPU also use 16 bytes per row
    // This makes writing 3x3 matrices to buffers annoying, since you need to pad each row with 4 bytes
    // https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
    // AlignOf(vec3<f32>) = AlignOf(vec4<f32>) = 16
    // SizeOf(mat3x3<f32>) = SizeOf(mat3x4<f32>) = 48
    /*const normMat3 = mat3.create();
    mat3.normalFromMat4(normMat3, modelMat);*/

    device.queue.writeBuffer(vsUniformBuffer, 0, modelMat);
    // device.queue.writeBuffer(vsUniformBuffer, 64, pvMat); 
    device.queue.writeBuffer(vsUniformBuffer, 128, normMat);
    // device.queue.writeBuffer(fsUniformBuffer, 0, cameraPos);
    // device.queue.writeBuffer(fsUniformBuffer, 12, lightPos);

    // Swap framebuffer
    colorTexture = context.getCurrentTexture();
    colorTextureView = colorTexture.createView();

    encodeCommands();

    window.requestAnimationFrame(render);
};
