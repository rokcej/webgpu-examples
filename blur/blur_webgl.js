import { mat4 } from "../lib//gl-matrix/esm/index.js";
import * as dat from "../lib/dat.gui/dat.gui.module.js"
import { coloredCube } from "../meshes.js";
import * as util from "../util.js";

// Context
const canvas = document.getElementById("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const gl = canvas.getContext("webgl2");

// Check WebGL 2 support
if (!gl) {
    document.body.innerHTML = "<div class=\"text\">Your browser does not support WebGL 2</div>";
    throw new Error("WebGL 2 not supported");
}

// Gaussian blur settings
const settings = {
    numPasses: 20,
    useRenderPipeline: true,
    useComputePipeline: false
}
// dat.gui
const gui = new dat.GUI();
gui.width = 320;
(() => { // Don't pollute the namespace
    const folder = gui.addFolder("Gaussian Blur");
    folder.add(settings, "numPasses", 0, 100).step(1).name("Number of Passes");
    folder.open();
})();

// Shaders
const vsSceneSource = `#version 300 es
in vec3 aPos;
in vec3 aColor;

uniform mat4 uPVMMat;

out vec3 vColor;

void main() {
    gl_Position = uPVMMat * vec4(aPos, 1.0);
    vColor = aColor;
}
`;
const fsSceneSource = `#version 300 es
precision highp float;
in vec3 vColor;

out vec4 oColor;

void main() {
    oColor = vec4(vColor, 1.0);
}
`;
const vsQuadSource = `#version 300 es
in vec2 aPos;

out vec2 vFragUV;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vFragUV = (aPos + vec2(1.0, 1.0)) * 0.5;
}
`;
const fsQuadSource = `#version 300 es
precision highp float;
in vec2 vFragUV;

uniform sampler2D uSampler;

out vec4 oColor;

void main() {
    oColor = texture(uSampler, vFragUV).rgba;
}
`;
const fsBlurSource = `#version 300 es
precision mediump float;

#define NUM_WEIGHTS 5
// http://dev.theomader.com/gaussian-kernel-calculator/
const float weights[NUM_WEIGHTS] = float[] (
    0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
); // Sigma = ~1.75, Kernel size = 11
    
in vec2 vFragUV;

uniform bool uHorizontal;
uniform sampler2D uSampler;

out vec4 oColor;

void main() {
	vec2 texOffset = 1.0 / vec2(textureSize(uSampler, 0)); // Gets size of single texel
	vec4 color = texture(uSampler, vFragUV) * weights[0]; // Current fragment's contribution
	if (uHorizontal) {
		for (int i = 1; i < NUM_WEIGHTS; ++i) {
			color += texture(uSampler, vFragUV + vec2(texOffset.x * float(i), 0.0)) * weights[i];
			color += texture(uSampler, vFragUV - vec2(texOffset.x * float(i), 0.0)) * weights[i];
		}
	} else {
		for (int i = 1; i < NUM_WEIGHTS; ++i) {
			color += texture(uSampler, vFragUV + vec2(0.0, texOffset.y * float(i))) * weights[i];
			color += texture(uSampler, vFragUV - vec2(0.0, texOffset.y * float(i))) * weights[i];
		}
	}
	oColor = color;
}
`;
let sceneProg = createProgram(gl, vsSceneSource, fsSceneSource);
let quadProg = createProgram(gl, vsQuadSource, fsQuadSource);
let blurProg = createProgram(gl, vsQuadSource, fsBlurSource);

let sceneAttribs = getAttributes(gl, sceneProg);
let sceneUniforms = getUniforms(gl, sceneProg);
let quadAttribs = getAttributes(gl, quadProg);
let quadUniforms = getAttributes(gl, quadProg);
// let blurAttribs = getAttributes(gl, blurProg); // Same as quadAttribs
let blurUniforms = getUniforms(gl, blurProg);


// Scene buffers
let sceneVbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, sceneVbo);
gl.bufferData(gl.ARRAY_BUFFER, coloredCube.vertices, gl.STATIC_DRAW);

let sceneVao = gl.createVertexArray();
gl.bindVertexArray(sceneVao);
gl.enableVertexAttribArray(sceneAttribs["aPos"]);
gl.vertexAttribPointer(sceneAttribs["aPos"], 3, gl.FLOAT, false, 6 * 4, 0);
gl.enableVertexAttribArray(sceneAttribs["aColor"]);
gl.vertexAttribPointer(sceneAttribs["aColor"], 3, gl.FLOAT, false, 6 * 4, 3 * 4);

let sceneEbo = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sceneEbo);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, coloredCube.indices, gl.STATIC_DRAW);

// Quad buffers
let quadVbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0
]), gl.STATIC_DRAW);

let quadVao = gl.createVertexArray();
gl.bindVertexArray(quadVao);
gl.enableVertexAttribArray(quadAttribs["aPos"]);
gl.vertexAttribPointer(quadAttribs["aPos"], 2, gl.FLOAT, false, 2 * 4, 0);

// Uniform data
const projMat = mat4.create();
const viewMat = mat4.create();
const pvMat = mat4.create();

mat4.perspective(projMat, Math.PI / 2, canvas.width / canvas.height, 0.1, 100.0);
mat4.lookAt(viewMat, [0, 0, 2], [0, 0, 0], [0, 1, 0]);
mat4.mul(pvMat, projMat, viewMat);

// Textures
let tex0 = gl.createTexture();
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, tex0);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

let tex1 = gl.createTexture();
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, tex1);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

// Framebuffers
let fb0 = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, fb0);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex0, 0);

let fb1 = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, fb1);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex1, 0);

gl.bindTexture(gl.TEXTURE_2D, null);


gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LEQUAL);
gl.enable(gl.CULL_FACE);

util.initFps();
render();


function render() {
    util.updateFps();
    // Update MVP matrix
    let t = performance.now() / 1000;

    const modelMat = mat4.create();
    mat4.rotateX(modelMat, modelMat, t * 0.75);
    mat4.rotateY(modelMat, modelMat, t * 1.5);
    const pvmMat = mat4.create();
    mat4.mul(pvmMat, pvMat, modelMat);


    // Render scene
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb0);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(sceneProg);
    gl.bindVertexArray(sceneVao);
    gl.uniformMatrix4fv(sceneUniforms["uPVMMat"], false, pvmMat);
    gl.drawElements(gl.TRIANGLES, coloredCube.indices.length, gl.UNSIGNED_SHORT, 0)

    // Apply Gaussian blur
    const _numPasses = settings.numPasses;
    for (let i = 0; i < _numPasses; ++i) {
        // Vertical pass
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb1);
        gl.bindTexture(gl.TEXTURE_2D, tex0);
        // gl.clearColor(0, 0, 0, 1);
        // gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(blurProg);
    
        gl.activeTexture(gl.TEXTURE0);
        gl.uniform1i(blurUniforms["uHorizontal"], false);
        gl.uniform1i(blurUniforms["uSampler"], 0);
        gl.bindVertexArray(quadVao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Horizontal pass
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb0);
        gl.bindTexture(gl.TEXTURE_2D, tex1);
        // gl.clearColor(0, 0, 0, 1);
        // gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
        gl.activeTexture(gl.TEXTURE0);
        gl.uniform1i(blurUniforms["uHorizontal"], true);
        gl.uniform1i(blurUniforms["uSampler"], 0);
        gl.bindVertexArray(quadVao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // Render fullscreen quad
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, tex0);
    // gl.clearColor(0, 0, 0, 1);
    // gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(quadProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(quadUniforms["uSampler"], 0);
    gl.bindVertexArray(quadVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);


    requestAnimationFrame(render);
}


function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.log(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        throw new Error("Can't compile shader");
    }
    return shader;
}
function createProgram(gl, vsSource, fsSource) {
    const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.log(gl.getProgramInfoLog(prog));
        gl.deleteProgram(prog);
        throw new Error("Can't create program");
    }
    return prog;
}
function getUniforms(gl, prog) {
    const uniforms = {};
	for (let i = 0; i < gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); ++i) {
		const info = gl.getActiveUniform(prog, i);
		uniforms[info.name] = gl.getUniformLocation(prog, info.name);
	}
    return uniforms;
}
function getAttributes(gl, prog) {
	const attribs = {};
	for (let i = 0; i < gl.getProgramParameter(prog, gl.ACTIVE_ATTRIBUTES); ++i) {
		const info = gl.getActiveAttrib(prog, i);
		attribs[info.name] = gl.getAttribLocation(prog, info.name);
	}
    return attribs;
}
