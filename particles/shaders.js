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


// https://stackoverflow.com/a/4275343
fn rand2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 4.1414)) % 3.14) * 43758.5453);
}


///////////////////////////////////////////////////////////
// 4D Simplex noise https://github.com/stegu/webgl-noise //
///////////////////////////////////////////////////////////

// fn mod4_289(x: vec4<f32>) -> vec4<f32> {
//     // return x % vec4<f32>(289.0);
//     return x - trunc(x * (1.0 / 289.0)) * 289.0;
// }
// fn mod_289(x: f32) -> f32 {
//     // return x % 289.0;
//     return x - trunc(x / 289.0) * 289.0;
// }

fn permute4(x: vec4<f32>) -> vec4<f32> {
    return (((x * 34.0) + 10.0) * x) % vec4<f32>(289.0);
}
fn permute(x: f32) -> f32 {
    return (((x * 34.0) + 10.0) * x) % 289.0;
}

fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}
fn taylorInvSqrt(r: f32) -> f32 {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn grad4(j: f32, ip: vec4<f32>) -> vec4<f32> {
    let ones = vec4<f32>(1.0, 1.0, 1.0, -1.0);

    var p = vec4<f32>(
        floor(fract(vec3<f32>(j) * ip.xyz) * 7.0) * ip.z - 1.0,
        0.0
    );
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    var s = vec4<f32>(p < vec4(0.0));
    p = vec4<f32>(
        p.xyz + (s.xyz*2.0 - 1.0) * s.www,
        p.w
    );

    return p;
}
                          
// (sqrt(5) - 1)/4 = F4, used once below
let F4: f32 = 0.309016994374947451;
  
fn simplexNoise(v: vec4<f32>) -> f32 {
    let C = vec4<f32>(
        0.138196601125011, // (5 - sqrt(5))/20 = G4
        0.276393202250021, // 2 * G4
        0.414589803375032, // 3 * G4
        -0.447213595499958 // -1 + 4 * G4
    );
    
    // First corner
    var i = floor(v + dot(v, vec4<f32>(F4)));
    var x0 = v - i + dot(i, C.xxxx);
  
    // Other corners
  
    // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
    var isX = step(x0.yzw, x0.xxx);
    var isYZ = step(x0.zww, x0.yyz);
    // i0.x = dot(isX, vec3(1.0));
    var i0 = vec4<f32>(
        isX.x + isX.y + isX.z,
        1.0 - isX
    );
    //  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
    i0.y = i0.y + isYZ.x + isYZ.y;
    //i0.zw = i0.zw + 1.0 - isYZ.xy;
    i0 = vec4<f32>(
        i0.xy,
        i0.zw + 1.0 - isYZ.xy,
    );
    i0.z = i0.z + isYZ.z;
    i0.w = i0.w + 1.0 - isYZ.z;
  
    // i0 now contains the unique values 0,1,2,3 in each channel
    var i3 = clamp(i0,       vec4<f32>(0.0), vec4<f32>(1.0));
    var i2 = clamp(i0 - 1.0, vec4<f32>(0.0), vec4<f32>(1.0));
    var i1 = clamp(i0 - 2.0, vec4<f32>(0.0), vec4<f32>(1.0));
  
    //  x0 = x0 - 0.0 + 0.0 * C.xxxx
    //  x1 = x0 - i1  + 1.0 * C.xxxx
    //  x2 = x0 - i2  + 2.0 * C.xxxx
    //  x3 = x0 - i3  + 3.0 * C.xxxx
    //  x4 = x0 - 1.0 + 4.0 * C.xxxx
    var x1 = x0 - i1 + C.xxxx;
    var x2 = x0 - i2 + C.yyyy;
    var x3 = x0 - i3 + C.zzzz;
    var x4 = x0 + C.wwww;
  
    // Permutations
    i = i % vec4<f32>(289.0);
    var j0 = permute(permute(permute(permute(i.w) + i.z) + i.y) + i.x);
    var j1 = permute4(permute4(permute4(permute4(
        i.w + vec4<f32>(i1.w, i2.w, i3.w, 1.0)) +
        i.z + vec4<f32>(i1.z, i2.z, i3.z, 1.0)) +
        i.y + vec4<f32>(i1.y, i2.y, i3.y, 1.0)) +
        i.x + vec4<f32>(i1.x, i2.x, i3.x, 1.0)
    );
  
    // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
    // 7*7*6 = 294, which is close to the ring size 17*17 = 289.
    var ip = vec4<f32>(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;
  
    var p0 = grad4(j0,   ip);
    var p1 = grad4(j1.x, ip);
    var p2 = grad4(j1.y, ip);
    var p3 = grad4(j1.z, ip);
    var p4 = grad4(j1.w, ip);
  
    // Normalise gradients
    var norm = taylorInvSqrt4(vec4<f32>(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;
    p4 = p4 * taylorInvSqrt(dot(p4,p4));
  
    // Mix contributions from the five corners
    var m0 = max(0.6 - vec3<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2)), vec3<f32>(0.0));
    var m1 = max(0.6 - vec2<f32>(dot(x3,x3), dot(x4,x4)), vec2<f32>(0.0));
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 49.0 * (
        dot(m0*m0, vec3(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
        dot(m1*m1, vec2(dot(p3, x3), dot(p4, x4)))
    );
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////


fn getPotential(p: vec4<f32>) -> vec3<f32> {
	return vec3<f32>(0.0);
}

fn getVelocity(pos: vec3<f32>, time: f32) -> vec3<f32> {
	return vec3<f32>(0.0);
}


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
