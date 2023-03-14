/**
 * References:
 * - https://alain.xyz/blog/raw-webgpu
 * - https://github.com/tsherif/webgpu-examples
 * - https://austin-eng.com/webgpu-samples/
 */

"use strict";

import { vec3, mat3, mat4 } from "../lib/gl-matrix/esm/index.js";
import { texturedCube } from "../meshes.js";
import { mainVertexSource, mainFragmentSource, shadowVertexSource } from "./shaders.js";

const shadowDepthTextureSize = 1024;

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
const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
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
const shadowDepthTexture = device.createTexture({
    size: [shadowDepthTextureSize, shadowDepthTextureSize, 1],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
});
const shadowDepthTextureView = shadowDepthTexture.createView();

let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus-stencil8",
    usage: GPUTextureUsage.RENDER_ATTACHMENT // | GPUTextureUsage.COPY_SRC
});
let depthTextureView = depthTexture.createView();
let colorTexture = context.getCurrentTexture();
let colorTextureView = colorTexture.createView();

// Buffers
const vertexBuffer = createBuffer(device, texturedCube.vertices, GPUBufferUsage.VERTEX);
const indexBuffer = createBuffer(device, texturedCube.indices, GPUBufferUsage.INDEX);
const indexCount = texturedCube.indices.length;


// Pipelines
const vertexBuffers = [
    {
        arrayStride: Float32Array.BYTES_PER_ELEMENT * 8,
        attributes: [
            {
                // Position
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
            },
            {
                // UV
                shaderLocation: 1,
                offset: Float32Array.BYTES_PER_ELEMENT * 3,
                format: 'float32x2',
            },
            {
                // Normal
                shaderLocation: 2,
                offset: Float32Array.BYTES_PER_ELEMENT * 5,
                format: 'float32x3',
            },
        ],
    },
];

const primitive = {
    topology: 'triangle-list',
    cullMode: 'back',
};

const uniformBufferBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: {
                type: 'uniform',
            },
        },
    ],
});

const shadowPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
        bindGroupLayouts: [
            uniformBufferBindGroupLayout,
            uniformBufferBindGroupLayout,
        ],
    }),
    vertex: {
        module: device.createShaderModule({
            code: shadowVertexSource,
        }),
        entryPoint: 'main',
        buffers: vertexBuffers,
    },
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth32float',
    },
    primitive,
});


const bglForRender = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: {
                type: 'uniform',
            },
        },
        {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            texture: {
                sampleType: 'depth',
            },
        },
        {
            binding: 2,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            sampler: {
                type: 'comparison',
            },
        },
    ],
});

const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
        bindGroupLayouts: [bglForRender, uniformBufferBindGroupLayout],
    }),
    vertex: {
        module: device.createShaderModule({
            code: mainVertexSource,
        }),
        entryPoint: 'main',
        buffers: vertexBuffers,
    },
    fragment: {
        module: device.createShaderModule({
            code: mainFragmentSource,
        }),
        entryPoint: 'main',
        targets: [
            {
                format: preferredFormat,
            },
        ],
        constants: {
            shadowDepthTextureSize,
        },
    },
    depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus-stencil8',
    },
    primitive,
});

const renderPassDescriptor = {
    colorAttachments: [
        {
            view: undefined, // Set in render loop

            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
        },
    ],
    depthStencilAttachment: {
        view: depthTexture.createView(),

        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        stencilClearValue: 0,
        stencilLoadOp: 'clear',
        stencilStoreOp: 'store',
    },
};

const modelUniformBuffer = device.createBuffer({
    size: 4 * 16, // 4x4 matrix
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const model2UniformBuffer = device.createBuffer({
    size: 4 * 16, // 4x4 matrix
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const sceneUniformBuffer = device.createBuffer({
    // Two 4x4 viewProj matrices,
    // one for the camera and one for the light.
    // Then a vec3 for the light position.
    // Rounded to the nearest multiple of 16.
    size: 2 * 4 * 16 + 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const sceneBindGroupForShadow = device.createBindGroup({
    layout: uniformBufferBindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: sceneUniformBuffer,
            },
        },
    ],
});

const sceneBindGroupForRender = device.createBindGroup({
    layout: bglForRender,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: sceneUniformBuffer,
            },
        },
        {
            binding: 1,
            resource: shadowDepthTextureView,
        },
        {
            binding: 2,
            resource: device.createSampler({
                compare: 'less',
            }),
        },
    ],
});

const modelBindGroup = device.createBindGroup({
    layout: uniformBufferBindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: modelUniformBuffer,
            },
        },
    ],
});

const model2BindGroup = device.createBindGroup({
    layout: uniformBufferBindGroupLayout,
    entries: [
        {
            binding: 0,
            resource: {
                buffer: model2UniformBuffer,
            },
        },
    ],
});


const shadowPassDescriptor = {
    colorAttachments: [],
    depthStencilAttachment: {
        view: shadowDepthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
    },
};


// Data
const eyePosition = vec3.fromValues(0, 5, -10);
const upVector = vec3.fromValues(0, 1, 0);
const origin = vec3.fromValues(0, 0, 0);

const projectionMatrix = mat4.create();
mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, canvas.width / canvas.height, 1, 2000.0);

const viewMatrix = mat4.create();
mat4.lookAt(viewMatrix, eyePosition, origin, upVector);

const lightPosition = vec3.fromValues(25, 50, -50);
const lightViewMatrix = mat4.create();
mat4.lookAt(lightViewMatrix, lightPosition, origin, upVector);

const lightProjectionMatrix = mat4.create();
{
    const left = -10;
    const right = 10;
    const bottom = -10;
    const top = 10;
    const near = -200;
    const far = 300;
    mat4.ortho(lightProjectionMatrix, left, right, bottom, top, near, far);
}

const lightViewProjMatrix = mat4.create();
mat4.multiply(lightViewProjMatrix, lightProjectionMatrix, lightViewMatrix);

const viewProjMatrix = mat4.create();
mat4.multiply(viewProjMatrix, projectionMatrix, viewMatrix);

// const modelMatrix = mat4.create();
// mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(0, 1, 0));
const model2Matrix = mat4.create();
mat4.scale(model2Matrix, model2Matrix, vec3.fromValues(10, 0.1, 10));

// The camera/light aren't moving, so write them into buffers now.
{
    const lightMatrixData = lightViewProjMatrix;
    device.queue.writeBuffer(
        sceneUniformBuffer,
        0,
        lightMatrixData.buffer,
        lightMatrixData.byteOffset,
        lightMatrixData.byteLength
    );

    const cameraMatrixData = viewProjMatrix;
    device.queue.writeBuffer(
        sceneUniformBuffer,
        64,
        cameraMatrixData.buffer,
        cameraMatrixData.byteOffset,
        cameraMatrixData.byteLength
    );

    const lightData = lightPosition;
    device.queue.writeBuffer(
        sceneUniformBuffer,
        128,
        lightData.buffer,
        lightData.byteOffset,
        lightData.byteLength
    );

    // const modelData = modelMatrix;
    // device.queue.writeBuffer(
    //     modelUniformBuffer,
    //     0,
    //     modelData.buffer,
    //     modelData.byteOffset,
    //     modelData.byteLength
    // );

    const model2Data = model2Matrix;
    device.queue.writeBuffer(
        model2UniformBuffer,
        0,
        model2Data.buffer,
        model2Data.byteOffset,
        model2Data.byteLength
    );
}

// Draw
render();


function getCameraViewProjMatrix() {
    const eyePosition = vec3.fromValues(0, 5, -10);

    const rad = 0.1 * Math.PI * (performance.now() / 1000);
    vec3.rotateY(eyePosition, eyePosition, origin, rad);

    const viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, eyePosition, origin, upVector);

    mat4.multiply(viewProjMatrix, projectionMatrix, viewMatrix);
    return viewProjMatrix;
}

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

    renderPass.drawIndexed(indexCount);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
}

function render() {
    // Camera
    const cameraViewProj = getCameraViewProjMatrix();
    device.queue.writeBuffer(
        sceneUniformBuffer,
        64,
        cameraViewProj.buffer,
        cameraViewProj.byteOffset,
        cameraViewProj.byteLength
    );

    // Cube
    const t = performance.now() / 1000;
    const modelMatrix = mat4.create();
    mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(0, 1, 0));
    mat4.rotateX(modelMatrix, modelMatrix, t * 0.75);
    mat4.rotateY(modelMatrix, modelMatrix, t * 1.5);
    device.queue.writeBuffer(
        modelUniformBuffer,
        0,
        modelMatrix.buffer,
        modelMatrix.byteOffset,
        modelMatrix.byteLength
    );

    // Render
    renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

    const commandEncoder = device.createCommandEncoder();
    {
        const shadowPass = commandEncoder.beginRenderPass(shadowPassDescriptor);
        shadowPass.setPipeline(shadowPipeline);
        shadowPass.setBindGroup(0, sceneBindGroupForShadow);
        shadowPass.setBindGroup(1, modelBindGroup);
        shadowPass.setVertexBuffer(0, vertexBuffer);
        shadowPass.setIndexBuffer(indexBuffer, 'uint16');
        shadowPass.drawIndexed(indexCount);

        shadowPass.setBindGroup(1, model2BindGroup);
        shadowPass.drawIndexed(indexCount);

        shadowPass.end();
    }
    {
        const renderPass = commandEncoder.beginRenderPass(renderPassDescriptor);
        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, sceneBindGroupForRender);
        renderPass.setBindGroup(1, modelBindGroup);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');
        renderPass.drawIndexed(indexCount);
        
        renderPass.setBindGroup(1, model2BindGroup);
        renderPass.drawIndexed(indexCount);

        renderPass.end();
    }
    device.queue.submit([commandEncoder.finish()]);
    window.requestAnimationFrame(render);
};
