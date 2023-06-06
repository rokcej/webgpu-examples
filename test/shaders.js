// Vertex and fragment shaders
export const renderSource = `
// Random number generation

var<private> xorshift32_state: u32 = 0u;
fn xorshift32() -> u32 {
    var s = xorshift32_state;
    s = s ^ (s << 13u);
    s = s ^ (s >> 17u);
    s = s ^ (s << 5u);
    xorshift32_state = s;
    return xorshift32_state;
}
fn rand() -> f32 {
    return f32(xorshift32()) / f32(~0u);
}
fn srand(seed: f32) {
    xorshift32_state = u32(seed * f32(~0u));
}


// Rendering

struct UBO {
    color: vec3<f32>,
    stepSize: f32
};
@group(0) @binding(0) var<uniform> ubo: UBO;

struct VSIn {
	@location(0) pos: vec2<f32>
};

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>
};

@vertex
fn vs_main(vsIn: VSIn) -> VSOut {
	var vsOut: VSOut;
	vsOut.pos = vec4<f32>(vsIn.pos.x, vsIn.pos.y, ubo.stepSize, 1.0);
    vsOut.uv = vsIn.pos * 0.5 + 0.5;
    return vsOut;
}

@fragment
fn fs_main(fsIn: VSOut) -> @location(0) vec4<f32> {
    var value: f32 = 0.0;

    srand(fsIn.uv.x * fsIn.uv.y);

    var t: f32 = 0.0;
    for (; t < 1.0;) {
        value = rand();
        t += ubo.stepSize;
    }

    return vec4<f32>(value, value, value, 1.0);
}
`;
