const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const canvas = document.getElementById("canvas");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

const context = canvas.getContext("webgpu");
context.configure({
    device,
    format: canvasFormat
});

const code = await fetch("shader.wgsl").then(response => response.text());
const module = device.createShaderModule({ code });

const bindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: "unfilterable-float" }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: "non-filtering" }
        }
    ]
});

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
});

const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
        module,
        entryPoint: "vertex_main",
    },
    fragment: {
        module,
        entryPoint: "fragment_main",
        targets: [{ format: canvasFormat }]
    }
});

const texture = device.createTexture({
    size: [2, 2],
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
})
device.queue.writeTexture(
    { texture },
    new Float32Array([
        1, 0, 0, 1,
        0, 1, 0, 1,
        0, 0, 1, 1,
        1, 1, 0, 1
    ]),
    {  bytesPerRow: 2 * 16 },
    [2, 2]
);

const sampler = device.createSampler(); 

const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: texture.createView()
        },
        {
            binding: 1,
            resource: sampler
        }
    ]
});

function render() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(),
                clearValue: [0, 0, 0, 1],
                loadOp: "clear",
                storeOp: "store"
            }
        ]
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(render);
}

requestAnimationFrame(render);
