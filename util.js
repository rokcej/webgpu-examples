const fpsCounter = {
	numFrames: 0,
	lastUpdated: 0,
	span: null
}

export function initFps() {
	let span = document.createElement("span");
	span.id = "fps";
	span.innerText = "0 FPS";
	document.body.appendChild(span);
	fpsCounter.span = span;
}

export function updateFps() {
	let t = performance.now() * 0.001;
	let dt = t - fpsCounter.lastUpdated;
	++fpsCounter.numFrames;
	if (dt > 0.5) {
		let fps = fpsCounter.numFrames / dt;
		fpsCounter.span.innerText = `${Math.round(fps)} FPS`;
		fpsCounter.numFrames = 0;
		fpsCounter.lastUpdated = t;
	}
}
