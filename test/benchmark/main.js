/*
- NUM_TESTS=10, NUM_RENDERS_PER_TEST=200000
    - 3 vertices
        - Average time: 11.619880000000075 seconds
        - Median time: 11.691699999999251 seconds
    - 6 vertices
        - Average time: 11.821079999999702 seconds
        - Median time: 11.836199999999252 seconds
*/

const NUM_TESTS = 10;
const NUM_RENDERS_PER_TEST = 200000

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const textureFormat = "rgba8unorm";
const texture = device.createTexture({
    size: [512, 512],
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    format: textureFormat
});
const textureView = texture.createView();

const code_3 = await fetch("shader_3.wgsl").then(response => response.text());
const code_6 = await fetch("shader_6.wgsl").then(response => response.text());
const module_3 = device.createShaderModule({ code: code_3 });
const module_6 = device.createShaderModule({ code: code_6 });

function createPipeline(module, format) {
    return device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vertex_main",
        },
        fragment: {
            module,
            entryPoint: "fragment_main",
            targets: [{ format }]
        }
    });
}
const pipeline_3 = createPipeline(module_3, textureFormat);
const pipeline_6 = createPipeline(module_6, textureFormat);

function render(pipeline, vertexCount) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: textureView,
                clearValue: [0, 0, 0, 1],
                loadOp: "clear",
                storeOp: "store"
            }
        ]
    });
    pass.setPipeline(pipeline);
    pass.draw(vertexCount);
    pass.end();

    device.queue.submit([encoder.finish()]);
}

function test(pipeline, vertexCount) {
    const time_start = performance.now() * 0.001;
    for (let i = 0; i < 200000; ++i) {
        render(pipeline, vertexCount);
    }
    const time_stop = performance.now() * 0.001;
    return time_stop - time_start;
}

function tests(pipeline, vertexCount) {
    const times = [];
    for (let i = 0; i < 10; ++i) {
        const time = test(pipeline, vertexCount);
        times.push(time);
    }

    const average = times.reduce((a, b) => a + b, 0) / times.length;
    const median = times.toSorted((a, b) => a - b)[Math.floor(times.length / 2)];

    console.log(`[Test results for ${vertexCount} vertices]`);
    console.log(`All times: ${times.join(", ")}`);
    console.log(`Average time: ${average}`);
    console.log(`Median time: ${median}`);
    console.log();
}

tests(pipeline_3, 3);
tests(pipeline_6, 6);
