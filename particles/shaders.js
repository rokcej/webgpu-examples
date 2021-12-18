// Vertex and fragment shaders
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
    let particleSize = 0.05;
	// http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/
	var worldPos = vsIn.pos +
		ubo.right * vsIn.quadPos.x * particleSize +
		ubo.up    * vsIn.quadPos.y * particleSize;

	var vsOut: VSOut;
	vsOut.pos = ubo.pvmMat * vec4<f32>(worldPos, 1.0);
	vsOut.color = vsIn.color;
    vsOut.quadPos = vsIn.quadPos;
    return vsOut;
}

[[stage(fragment)]]
fn fs_main(fsIn: VSOut) -> [[location(0)]] vec4<f32> {
	var color = fsIn.color;
    color.a = color.a * smoothStep(1.0, 0.0, length(fsIn.quadPos));
    return color;
}
`;

// Compute shader
export const computeSource = `
struct UBO {
	seed: f32;
	deltaTime: f32;
    time: f32;
    flowScale: f32;
	flowEvolution: f32;
    flowSpeed: f32;
    artisticMode: u32;
	numParticles: u32;
};

struct Particle {
	position: vec3<f32>;
	life: f32;
	color: vec4<f32>;
    age: f32;
    artisticMode: u32;
};

struct Data {
	particles: array<Particle>;
};

[[group(0), binding(0)]] var<uniform> ubo: UBO;
[[group(0), binding(1)]] var<storage, read_write> data: Data;


let PI: f32 = 3.1415926535897932384626433;


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
    var s = vec4<f32>(p < vec4<f32>(0.0));
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
    // i0.x = dot(isX, vec3<f32>(1.0));
    var i0 = vec4<f32>(
        isX.x + isX.y + isX.z,
        1.0 - isX
    );
    //  i0.y += dot( isYZ.xy, vec2<f32>( 1.0 ) );
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
        dot(m0*m0, vec3<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
        dot(m1*m1, vec2<f32>(dot(p3, x3), dot(p4, x4)))
    );
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////


// Curl noise
fn getPotential(p: vec4<f32>) -> vec3<f32> {
    // Simplex noise sampled at large offsets
	return vec3<f32>(
        simplexNoise(p),
        simplexNoise(vec4<f32>(p.z - 9.61, p.x + 3.14, p.y + 9.77, p.w)),
		simplexNoise(vec4<f32>(p.y - 2.32, p.z + 12.7, p.x + 11.97, p.w))
    );
}
fn getVelocity(pos: vec3<f32>, time: f32) -> vec3<f32> {
	var p = vec4<f32>(pos / ubo.flowScale, time * ubo.flowEvolution);
	var pot = getPotential(p);

	var eps = 0.0001;
	var dx = vec4<f32>(eps, 0.0, 0.0, 0.0);
	var dy = vec4<f32>(0.0, eps, 0.0, 0.0);
	var dz = vec4<f32>(0.0, 0.0, eps, 0.0);

	// Partial derivatives
	var dp_dx = getPotential(p + dx);
	var dp_dy = getPotential(p + dy);
	var dp_dz = getPotential(p + dz);

    var invEps = 1.0 / eps;
	var dp3_dy = (dp_dy.z - pot.z) * invEps;
	var dp2_dz = (dp_dz.y - pot.y) * invEps;
	var dp1_dz = (dp_dz.x - pot.x) * invEps;
	var dp3_dx = (dp_dx.z - pot.z) * invEps;
	var dp2_dx = (dp_dx.y - pot.y) * invEps;
	var dp1_dy = (dp_dy.x - pot.x) * invEps;

	return vec3<f32>(dp3_dy - dp2_dz, dp1_dz - dp3_dx, dp2_dx - dp1_dy) * ubo.flowSpeed;
}


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

// Uniform unit sphere sampling
fn sampleUnitSphere() -> vec3<f32> {
	// Uniform spherical coordinates
	var u   = rand(); // u ∈ [0, 1]
	var v   = rand() * 2.0 - 1.0; // v ∈ [-1, 1]
	var phi = rand() * 2.0 * PI; // φ ∈ [0, 2π)

	var r = pow(u, 1.0 / 3.0); // r ∈ [0, 1]
	var cosTheta = -v; // θ ∈ [0, π]
	var sinTheta = sqrt(1.0 - v * v);

	return vec3<f32>(
		r * sinTheta * cos(phi),
		r * sinTheta * sin(phi),
		r * cosTheta
	);
}
fn sampleUnitSphereSurface() -> vec3<f32> {
	// Uniform sampling of unit sphere surface
	// http://corysimon.github.io/articles/uniformdistn-on-sphere/
	var cosTheta = 1.0 - 2.0 * rand();
	var sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	var phi = 2.0 * PI * rand();
	return vec3<f32>(
		sinTheta * cos(phi),
		sinTheta * sin(phi),
		cosTheta
	);
}


[[stage(compute), workgroup_size(256)]]
fn main([[builtin(global_invocation_id)]] gid: vec3<u32>) {
	if (gid.x >= ubo.numParticles) {
        return;
    }

    let spawnRadius = 60.0;
    let spawnRadiusArtistic = 25.0;

	var particle = data.particles[gid.x]; // Read particle data

    if (ubo.artisticMode == particle.artisticMode) {
        particle.life = particle.life - ubo.deltaTime;
    } else {
        particle.life = 0.0;
    }

    if (particle.life > 0.0) {
        var vel = getVelocity(particle.position, ubo.time);

        if (!bool(ubo.artisticMode)) {
            particle.position = particle.position + vel * ubo.deltaTime;
        } else {
            particle.position = particle.position + vel * ubo.deltaTime * 2.0;
        }

        particle.age = particle.age + ubo.deltaTime;
        particle.color = vec4<f32>(
            (abs(vel) / (2.0 * ubo.flowSpeed)),
            min(particle.life, min(particle.age * 4.0, 1.0))
        );
    } else {
        srand((ubo.seed + f32(gid.x) / f32(ubo.numParticles)) * 0.5); // Init RNG

        if (!bool(ubo.artisticMode)) {
            particle.position = sampleUnitSphere() * spawnRadius;
            particle.life = rand() * 15.0 + 15.0;
        } else {
            particle.position = sampleUnitSphereSurface() * spawnRadiusArtistic + 
                (vec3<f32>(rand(), rand(), rand()) - 0.5); // Add noise to get rid of artifacts
            particle.life = rand() * 3.0 + 3.0;
        }

        particle.artisticMode = ubo.artisticMode;
        particle.age = 0.0;
        particle.color = vec4<f32>(0.0);
    }

	data.particles[gid.x] = particle; // Write particle data
}
`;
