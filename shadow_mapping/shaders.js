export const mainVertexSource = `
struct Scene {
    lightViewProjMatrix: mat4x4<f32>,
    cameraViewProjMatrix: mat4x4<f32>,
    lightPos: vec3<f32>,
}

struct Model {
    modelMatrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(1) @binding(0) var<uniform> model : Model;

struct VertexOutput {
    @location(0) shadowPos: vec3<f32>,
    @location(1) fragPos: vec3<f32>,
    @location(2) fragNorm: vec3<f32>,

    @builtin(position) Position: vec4<f32>,
}

@vertex
fn main(
    @location(0) position: vec3<f32>,
    @location(1) uv: vec3<f32>,
    @location(2) normal: vec3<f32>
) -> VertexOutput {
    var output : VertexOutput;

    // XY is in (-1, 1) space, Z is in (0, 1) space
    let posFromLight = scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);

    // Convert XY to (0, 1)
    // Y is flipped because texture coords are Y-down.
    output.shadowPos = vec3(
        posFromLight.xy * vec2(0.5, -0.5) + vec2(0.5),
        posFromLight.z
    );

    output.Position = scene.cameraViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
    output.fragPos = output.Position.xyz;
    output.fragNorm = normal;
    return output;
}
`;

export const mainFragmentSource = `
override shadowDepthTextureSize: f32 = 1024.0;

struct Scene {
    lightViewProjMatrix : mat4x4<f32>,
    cameraViewProjMatrix : mat4x4<f32>,
    lightPos : vec3<f32>,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;

struct FragmentInput {
    @location(0) shadowPos : vec3<f32>,
    @location(1) fragPos : vec3<f32>,
    @location(2) fragNorm : vec3<f32>,
}

const albedo = vec3<f32>(0.9);
const ambientFactor = 0.2;

@fragment
fn main(input : FragmentInput) -> @location(0) vec4<f32> {
    // Percentage-closer filtering. Sample texels in the region
    // to smooth the result.
    var visibility = 0.0;
    let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
    for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
        let offset = vec2<f32>(vec2(x, y)) * oneOverShadowDepthTextureSize;

        visibility += textureSampleCompare(
        shadowMap, shadowSampler,
        input.shadowPos.xy + offset, input.shadowPos.z - 0.007
        );
    }
    }
    visibility /= 9.0;

    let lambertFactor = max(dot(normalize(scene.lightPos - input.fragPos), input.fragNorm), 0.0);
    let lightingFactor = min(ambientFactor + visibility * lambertFactor, 1.0);

    return vec4(lightingFactor * albedo, 1.0);
}
`;

export const shadowVertexSource = `
struct Scene {
    lightViewProjMatrix: mat4x4<f32>,
    cameraViewProjMatrix: mat4x4<f32>,
    lightPos: vec3<f32>,
}

struct Model {
    modelMatrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> scene : Scene;
@group(1) @binding(0) var<uniform> model : Model;

@vertex
fn main(
    @location(0) position: vec3<f32>
) -> @builtin(position) vec4<f32> {
    return scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
}
`;
