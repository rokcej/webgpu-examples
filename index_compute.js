/**
 * By Žiga Lesar
 */

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// we do not need this for now
//const canvas = document.querySelector('canvas');
//const context = canvas.getContext('webgpu');
//const format = context.getPreferredFormat(adapter);
//const swapchain = context.configure({ device, format });
//const texture = context.getCurrentTexture();

const workgroupSize = 256;
const code = `
[[block]]
struct ComputeInput {
    size : u32;
    data : array<f32>;
};

[[group(0), binding(0)]]
var<storage, read_write> buffer : ComputeInput;

[[stage(compute), workgroup_size(${workgroupSize})]]
fn main([[builtin(global_invocation_id)]] gid : vec3<u32>) {
    let index : u32 = gid.x;
    if (index < buffer.size) {
        buffer.data[index] = buffer.data[index] * buffer.data[index];
    }
}
`;

const bindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
            },
        }
    ],
});

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [ bindGroupLayout ]
});
const shader = device.createShaderModule({ code });
const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
        module: shader,
        entryPoint: 'main',
    }
});

// Breaks down at 1 << 25, because it is 128 MiB + 4 B, which is 4 B over the SSBO limit.
// Also, when using large buffer sizes, max thread count cannot reach all elements, and we need a loop inside the shader.
// Or, we could use dynamic buffer offsets, or a uniform that stores the offset/batchIndex/...
const bufferElements = 1 << 20;
const bufferSize = Uint32Array.BYTES_PER_ELEMENT + bufferElements * Float32Array.BYTES_PER_ELEMENT;

// prepare data
const writeBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
});

const writeData = writeBuffer.getMappedRange();
const writeDataSize = new Uint32Array(writeData, 0, 1);
const writeDataElements = new Float32Array(writeData, Uint32Array.BYTES_PER_ELEMENT, bufferElements);

writeDataSize[0] = bufferElements;
for (let i = 0; i < bufferElements; i++) {
    writeDataElements[i] = i;
}
writeBuffer.unmap();

// prepare read buffer
const readBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

// execute compute module
const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    //layout: pipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: {
                buffer: writeBuffer,
            },
        },
    ],
});
const computeEncoder = device.createCommandEncoder();
const computePass = computeEncoder.beginComputePass();
computePass.setPipeline(pipeline);
computePass.setBindGroup(0, bindGroup);
computePass.dispatch(Math.ceil(bufferElements / workgroupSize));
computePass.endPass();
const computeCommands = computeEncoder.finish();
device.queue.submit([computeCommands]);

// copy data
const copyEncoder = device.createCommandEncoder();
copyEncoder.copyBufferToBuffer(
    writeBuffer, 0,
    readBuffer, 0,
    bufferSize
);
const copyCommands = copyEncoder.finish();
device.queue.submit([copyCommands]);

// read back data
await readBuffer.mapAsync(GPUMapMode.READ);
const readData = new Float32Array(readBuffer.getMappedRange().slice(4)); // remove length
readBuffer.unmap();

console.log(readData);
