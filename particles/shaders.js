export const renderSource = `
struct UBO {
    pvmMat: mat4x4<f32>;
	right: vec3<f32>;
	up: vec3<f32>;
};
[[group(0), binding(0)]] var<uniform> ubo: UBO;

struct VSIn {
	[[location(0)]] pos: vec3<f32>;
    [[location(1)]] color: vec4<f32>;
	[[location(2)]] quadPos: vec2<f32>;
};

struct VSOut {
    [[builtin(position)]] pos: vec4<f32>;
    [[location(0)]] color: vec4<f32>;
	[[location(1)]] quadPos: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(vsIn: VSIn) -> VSOut {
	// http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/
	var worldPos = vsIn.pos +
		ubo.right * vsIn.quadPos.x * 0.05 +
		ubo.up    * vsIn.quadPos.y * 0.05;

	var vsOut: VSOut;
	vsOut.pos = ubo.pvmMat * vec4<f32>(worldPos, 1.0);
	vsOut.color = vsIn.color;
    vsOut.quadPos = vsIn.quadPos;
    return vsOut;
}

[[stage(fragment)]]
fn fs_main(fsIn: VSOut) -> [[location(0)]] vec4<f32> {
	var color = fsIn.color;
	color.a = color.a * max(1.0 - length(fsIn.quadPos), 0.0);
    return color;
}
`;
export const computeSource = `
struct UBO {
	seed: vec4<f32>;
	deltaTime: f32;
	numParticles: u32;
};

struct Particle {
	position: vec3<f32>;
	lifetime: f32;
	color: vec4<f32>;
	velocity: vec3<f32>;
};

struct Data {
	particles: array<Particle>;
};

[[group(0), binding(0)]] var<uniform> ubo: UBO;
[[group(0), binding(1)]] var<storage, read_write> data: Data;

[[stage(compute), workgroup_size(256)]]
fn main([[builtin(global_invocation_id)]] gid: vec3<u32>) {
	if (gid.x >= ubo.numParticles) {
        return;
    }

	var particle = data.particles[gid.x];
	particle.velocity.y = particle.velocity.y - 3.14 * ubo.deltaTime;
	particle.position = particle.position + particle.velocity * ubo.deltaTime;
	data.particles[gid.x] = particle;
}
`;
