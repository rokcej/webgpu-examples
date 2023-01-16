const fsBlurSource = `
// Gaussian blur
// https://learnopengl.com/Advanced-Lighting/Bloom

struct UBO {
    horizontal: u32
};
@group(0) @binding(0) var<uniform> ubo: UBO;
@group(0) @binding(1) var uSampler: sampler;
@group(0) @binding(2) var uTexture: texture_2d<f32>;

@fragment
fn main(@location(0) fragUV: vec2<f32>) -> @location(0) vec4<f32> {
    var weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    var uvStep = 1.0 / vec2<f32>(textureDimensions(uTexture));
    var result = textureSample(uTexture, uSampler, fragUV) * weights[0];
    if (ubo.horizontal != 0u) {
        for (var i: i32 = 1; i < 5; i = i + 1) {
            result = result + textureSample(uTexture, uSampler, fragUV + vec2<f32>(uvStep.x * f32(i), 0.0)) * weights[i];
            result = result + textureSample(uTexture, uSampler, fragUV - vec2<f32>(uvStep.x * f32(i), 0.0)) * weights[i];
        }
    } else {
        for (var i: i32 = 1; i < 5; i = i + 1) {
            result = result + textureSample(uTexture, uSampler, fragUV + vec2<f32>(0.0, uvStep.y * f32(i))) * weights[i];
            result = result + textureSample(uTexture, uSampler, fragUV - vec2<f32>(0.0, uvStep.y * f32(i))) * weights[i];
        }
    }
    return result;
}
`;

export class GaussianBlurRender {
    constructor(_device, _size, _inputTexture, _inputTextureView, _vsQuadModule) {
        // Save parameters
        this.device = _device;
        this.size = _size;
        this.inputTexture = _inputTexture;
        this.inputTextureView = _inputTextureView;
        this.vsQuadModule = _vsQuadModule;

        // Sampler
        this.sampler = this.device.createSampler({
            minFilter: "nearest",
            magFilter: "nearest"
        });

        // Textures
        this.blurTextures = [0, 1].map(() => {
            return this.device.createTexture({
                size: this.size,
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
            });
        });
        this.blurTextureViews = this.blurTextures.map(tex => tex.createView());

        // Shaders
        this.fsBlurModule = this.device.createShaderModule({ code: fsBlurSource });

        // Uniforms
        this.blurBuffers = [0, 1].map((val) => {
            const buffer = this.device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.UNIFORM,
                mappedAtCreation: true
            });
            new Uint32Array(buffer.getMappedRange())[0] = val;
            buffer.unmap();
            return buffer;
        });

        // Bind groups
        this.blurBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: "filtering" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: "float" }
                }
            ]
        });
        this.blurBindGroup0a = this.device.createBindGroup({ // Vertical pass (initial)
            layout: this.blurBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.blurBuffers[0] }
                },
                {
                    binding: 1,
                    resource: this.sampler
                },
                {
                    binding: 2,
                    resource: this.inputTextureView
                }
            ]
        });
        this.blurBindGroup0b = this.device.createBindGroup({ // Vertical pass
            layout: this.blurBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.blurBuffers[0] }
                },
                {
                    binding: 1,
                    resource: this.sampler
                },
                {
                    binding: 2,
                    resource: this.blurTextureViews[1]
                }
            ]
        });
        this.blurBindGroup1 = this.device.createBindGroup({ // Horizontal pass
            layout: this.blurBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.blurBuffers[1] }
                },
                {
                    binding: 1,
                    resource: this.sampler
                },
                {
                    binding: 2,
                    resource: this.blurTextureViews[0]
                }
            ]
        });

        // Render pipeline
        this.blurPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.blurBindGroupLayout]
            }),
            // Vertex shader
            vertex: {
                module: this.vsQuadModule,
                entryPoint: "main",
            },
            // Fragment shader
            fragment: {
                module: this.fsBlurModule,
                entryPoint: "main",
                targets: [{ format: "rgba8unorm" }],
            },
            // Rasterization
            primitive: { topology: "triangle-list" }
        });
    }

    render(commandEncoder, numPasses) {
        if (numPasses == 0) {
            commandEncoder.copyTextureToTexture(
                { texture: this.inputTexture },
                { texture: this.blurTextures[1] },
                this.size
            );
        } else {
            for (let i = 0; i < numPasses; ++i) {
                // Vertical pass
                const blurPass0 = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: this.blurTextureViews[0],
                        clearValue: [0, 0, 0, 1],
                        loadOp: "clear",
                        storeOp: "store"
                    }]
                });
                blurPass0.setPipeline(this.blurPipeline);
                blurPass0.setBindGroup(0, i == 0 ? this.blurBindGroup0a : this.blurBindGroup0b);
                blurPass0.draw(6, 1, 0, 0);
                blurPass0.end();
                // Horizontal pass
                const blurPass1 = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: this.blurTextureViews[1],
                        clearValue: [0, 0, 0, 1],
                        loadOp: "clear",
                        storeOp: "store"
                    }]
                });
                blurPass1.setPipeline(this.blurPipeline);
                blurPass1.setBindGroup(0, this.blurBindGroup1);
                blurPass1.draw(6, 1, 0, 0);
                blurPass1.end();
            }
        }
    }

    getOutputTexture() {
        return this.blurTextures[1];
    }

    getOutputTextureView() {
        return this.blurTextureViews[1];
    }
}
