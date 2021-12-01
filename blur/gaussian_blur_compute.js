const WORKGROUP_SIZE = [16, 16];

const csBlurSource = `
// Gaussian blur
// https://learnopengl.com/Advanced-Lighting/Bloom

[[block]] struct UBO {
    horizontal: u32;
};
[[group(0), binding(0)]] var<uniform> ubo: UBO;
[[group(0), binding(1)]] var uSampler: sampler;
[[group(0), binding(2)]] var uInTexture: texture_2d<f32>;
[[group(0), binding(3)]] var uOutTexture : texture_storage_2d<rgba8unorm, write>;

[[stage(compute), workgroup_size(${WORKGROUP_SIZE[0]}, ${WORKGROUP_SIZE[1]})]]
fn main([[builtin(global_invocation_id)]] gid : vec3<u32>) {
    var texDim: vec2<i32> = textureDimensions(uInTexture);

    if (gid.x >= u32(texDim.x) || gid.y >= u32(texDim.y)) {
        return;
    }

    var weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    var uvStep: vec2<f32> = 1.0 / vec2<f32>(texDim);
    var uv: vec2<f32> = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5) * uvStep;

    var result = textureSampleLevel(uInTexture, uSampler, uv, 0.0) * weights[0];
    if (ubo.horizontal != 0u) {
        for (var i: i32 = 1; i < 5; i = i + 1) {
            result = result + textureSampleLevel(uInTexture, uSampler, uv + vec2<f32>(uvStep.x * f32(i), 0.0), 0.0) * weights[i];
            result = result + textureSampleLevel(uInTexture, uSampler, uv - vec2<f32>(uvStep.x * f32(i), 0.0), 0.0) * weights[i];
        }
    } else {
        for (var i: i32 = 1; i < 5; i = i + 1) {
            result = result + textureSampleLevel(uInTexture, uSampler, uv + vec2<f32>(0.0, uvStep.y * f32(i)), 0.0) * weights[i];
            result = result + textureSampleLevel(uInTexture, uSampler, uv - vec2<f32>(0.0, uvStep.y * f32(i)), 0.0) * weights[i];
        }
    }
    textureStore(uOutTexture, vec2<i32>(i32(gid.x), i32(gid.y)), result);
}
`;

export class GaussianBlurCompute {
    constructor(_device, _size, _inputTexture, _inputTextureView) {
        // Save parameters
        this.device = _device;
        this.size = _size;
        this.inputTexture = _inputTexture;
        this.inputTextureView = _inputTextureView;

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
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING
            });
        });
        this.blurTextureViews = this.blurTextures.map(tex => tex.createView());

        // Shaders
        this.csBlurModule = this.device.createShaderModule({ code: csBlurSource });

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
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: { type: "filtering" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "float" }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: "write-only", 
                        format: "rgba8unorm"
                    }
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
                },
                {
                    binding: 3,
                    resource: this.blurTextureViews[1]
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
                    resource: this.blurTextureViews[0]
                },
                {
                    binding: 3,
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
                    resource: this.blurTextureViews[1]
                },
                {
                    binding: 3,
                    resource: this.blurTextureViews[0]
                }
            ]
        });
        
        // Compute pipeline
        this.blurPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.blurBindGroupLayout]
            }),
            // Compute shader
            compute: {
                module: this.csBlurModule,
                entryPoint: "main",
            }
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
            const numWorkgroups = [
                Math.ceil(this.size[0] / WORKGROUP_SIZE[0]),
                Math.ceil(this.size[1] / WORKGROUP_SIZE[1])
            ]
            const blurPass = commandEncoder.beginComputePass();
            blurPass.setPipeline(this.blurPipeline);
            for (let i = 0; i < numPasses; ++i) {
                // Vertical pass
                blurPass.setBindGroup(0, i == 0 ? this.blurBindGroup0a : this.blurBindGroup0b);
                blurPass.dispatch(numWorkgroups[0], numWorkgroups[1]);
                // Horizontal pass
                blurPass.setBindGroup(0, this.blurBindGroup1);
                blurPass.dispatch(numWorkgroups[0], numWorkgroups[1]);
            }
            blurPass.endPass();
        }
    }

    getOutputTexture() {
        return this.blurTextures[1];
    }

    getOutputTextureView() {
        return this.blurTextureViews[1];
    }
}
