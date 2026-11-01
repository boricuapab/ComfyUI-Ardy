import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';
import { api } from '../../../scripts/api.js';
import { queuePreview } from './queue.js';

export function pickFile(accept) {
    return new Promise(resolve => {
        const input = document.createElement('input');
        input.type = 'file'; input.accept = accept; input.hidden = true;
        document.body.append(input);
        input.onchange = () => { const file = input.files[0]; input.remove(); resolve(file); };
        input.oncancel = () => { input.remove(); resolve(null); };
        input.click();
    });
}

export class CameraPreview {
    constructor(node) {
        this.node = node;
        this.time = 0;
        this.playing = false;
        this.recording = false;
        this.track = [];
        this.root = document.createElement('div');
        this.root.style.cssText = 'height:540px;width:100%;display:flex;flex-direction:column;background:#171d25;color:#e5edf5;font:12px sans-serif;overflow:hidden;border-radius:8px';
        this.viewport = document.createElement('div');
        this.viewport.style.cssText = 'flex:1;min-height:260px;display:flex;align-items:center;justify-content:center;overflow:hidden';
        this.root.append(this.viewport);
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color('#000000');
        this.camera = new THREE.PerspectiveCamera(40, 1, .01, 10000);
        this.renderer = new THREE.WebGLRenderer({antialias: true});
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.viewport.append(this.renderer.domElement);
        this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
        this.orbit.mouseButtons.RIGHT = THREE.MOUSE.PAN;
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x555555, 2));
        const light = new THREE.DirectionalLight(0xffffff, 2);
        light.position.set(3, 8, 5); this.scene.add(light);
        this.grid = new THREE.GridHelper(200, 200, 0x697d91, 0x354452);
        this.scene.add(this.grid);
        const bar = document.createElement('div');
        bar.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;padding:8px;flex-shrink:0';
        this.root.append(bar);
        const button = (label, action) => {
            const el = document.createElement('button');
            el.textContent = label;
            el.style.cssText = 'background:#34485d;color:white;border:0;border-radius:4px;padding:7px 10px;cursor:pointer';
            el.onclick = action; bar.append(el); return el;
        };
        button('Load / render frames', async () => {
            this.finishRecording();
            this.status.textContent = 'Rendering saved take…';
            try { await queuePreview(node); }
            catch (error) { this.status.textContent = error.message; }
        });
        this.play = button('Play', () => {
            if (!this.data) return;
            if (this.time >= this.duration) this.time = 0;
            this.playing = !this.playing;
            this.play.textContent = this.playing ? 'Pause' : 'Play';
        });
        this.record = button('Record camera', () => {
            if (!this.data) return;
            this.time = 0; this.track = []; this.recording = true; this.playing = true;
            this.capture(); this.play.textContent = 'Pause';
            this.status.textContent = 'Recording camera — drag to orbit, right-drag to pan, scroll to zoom.';
        });
        button('Stop camera', () => { this.finishRecording(); this.playing = false; this.play.textContent = 'Play'; });
        button('Use current camera', () => {
            this.recording = false; this.track = [this.cameraKey(0)]; this.saveTrack();
            this.status.textContent = 'Static camera saved. Load / render frames to apply.';
        });
        button('Frame motion', () => this.frameMotion());
        button('Load NPZ…', async () => {
            const file = await pickFile('.npz');
            if (!file) return;
            try {
                const form = new FormData(); form.append('file', file);
                const response = await api.fetchApi('/ardy/recording/upload', {method: 'POST', body: form});
                if (!response.ok) throw new Error(await response.text());
                node.widgets.find(w => w.name === 'recording_id').value = (await response.json()).recording_id;
                node.graph?.change(); this.status.textContent = 'Loading saved take…';
                await queuePreview(node);
            } catch (error) { this.status.textContent = error.message; }
        });
        button('Use connected motion', () => {
            node.widgets.find(w => w.name === 'recording_id').value = '';
            node.graph?.change(); this.status.textContent = 'Next render uses connected motion.';
        });
        button('Save camera JSON', () => {
            this.finishRecording();
            if (!this.track.length) { this.status.textContent = 'Record a camera or use current camera first.'; return; }
            const url = URL.createObjectURL(new Blob([JSON.stringify(this.track, null, 2)], {type: 'application/json'}));
            const link = document.createElement('a'); link.href = url; link.download = 'ardy_camera.json'; link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        });
        button('Load camera JSON…', async () => {
            const file = await pickFile('.json');
            if (!file) return;
            try {
                const track = JSON.parse(await file.text());
                if (!Array.isArray(track) || !track.length) throw new Error('Choose a saved ARDY camera track.');
                this.playing = false; this.recording = false; this.track = track;
                this.saveTrack(); this.time = 0; this.applyTrack();
                this.status.textContent = 'Camera loaded. Load / render frames to apply.';
            } catch (error) { this.status.textContent = error.message; }
        });
        const label = document.createElement('label');
        const grid = document.createElement('input'); grid.type = 'checkbox'; grid.checked = true;
        grid.onchange = () => { this.grid.visible = grid.checked; };
        label.append(grid, ' Grid (guide only)'); bar.append(label);
        this.scrub = document.createElement('input'); this.scrub.type = 'range'; this.scrub.min = 0;
        this.scrub.max = 1; this.scrub.step = .001; this.scrub.value = 0;
        this.scrub.setAttribute('aria-label', 'Camera preview time');
        this.scrub.oninput = () => {
            this.finishRecording(); this.playing = false; this.play.textContent = 'Play';
            this.time = Number(this.scrub.value); this.updatePose(); this.applyTrack();
        };
        this.root.append(this.scrub);
        this.status = document.createElement('div'); this.status.style.cssText = 'padding:8px;min-height:16px';
        this.status.textContent = 'Load the saved take, then frame it or record a camera move.';
        this.root.append(this.status);
        for (const event of ['pointerdown', 'pointerup', 'wheel']) this.root.addEventListener(event, e => e.stopPropagation());
        this.root.addEventListener('contextmenu', e => { e.preventDefault(); e.stopPropagation(); });
        this.onWheel = e => {
            if (!this.viewport.contains(e.target)) return;
            e.preventDefault(); e.stopPropagation();
            this.previewTrack = false;
            const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? this.viewport.clientHeight : 1);
            const offset = this.camera.position.clone().sub(this.orbit.target);
            offset.multiplyScalar(Math.exp(Math.max(-1, Math.min(1, delta * .001))));
            this.camera.position.copy(this.orbit.target).add(offset); this.orbit.update();
        };
        window.addEventListener('wheel', this.onWheel, {capture: true, passive: false});
        this.orbit.addEventListener('start', () => { this.previewTrack = false; });
        this.resize = new ResizeObserver(() => this.resizeViewport()); this.resize.observe(this.viewport);
        this.last = performance.now();
        const animate = now => {
            this.frame = requestAnimationFrame(animate);
            const dt = Math.min((now - this.last) / 1000, .1); this.last = now;
            if (this.data && this.playing) {
                this.time = Math.min(this.time + dt, this.duration);
                this.updatePose();
                if (this.recording) {
                    if (this.time - this.track.at(-1).time >= 1 / this.data.fps || this.time === this.duration) this.capture();
                } else if (this.previewTrack) this.applyTrack();
                if (this.time >= this.duration) {
                    this.finishRecording(); this.playing = false; this.play.textContent = 'Play';
                }
            }
            this.renderer.render(this.scene, this.camera);
        };
        this.frame = requestAnimationFrame(animate);
    }

    value(name) { return this.node.widgets.find(w => w.name === name)?.value; }

    async load(file) {
        const request = this.request = (this.request ?? 0) + 1;
        const response = await api.fetchApi(`/view?${new URLSearchParams(file)}`);
        if (!response.ok) throw new Error('Saved preview is missing. Load / render frames again.');
        const buffer = await response.arrayBuffer();
        if (this.disposed || request !== this.request) return;
        const size = new DataView(buffer).getUint32(0, true);
        this.data = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, size)));
        const faces = new Uint32Array(buffer, 4 + size, this.data.faces);
        this.vertices = new Float32Array(buffer, 4 + size + faces.byteLength);
        if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
        const geometry = new THREE.BufferGeometry();
        geometry.setIndex(new THREE.BufferAttribute(faces, 1));
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.data.vertices * 3), 3).setUsage(THREE.DynamicDrawUsage));
        this.mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: 0xcccccc, roughness: 1, side: THREE.DoubleSide}));
        this.mesh.frustumCulled = false; this.scene.add(this.mesh);
        this.duration = Math.max(0, (this.data.frames - 1) / this.data.fps);
        this.scrub.max = this.duration; this.time = 0;
        this.scene.background.set(this.value('background'));
        this.track = this.data.camera_track?.length ? this.data.camera_track : JSON.parse(this.value('camera_track')?.trim() || '[]');
        this.previewTrack = this.track.length > 0;
        this.updatePose(); this.resizeViewport();
        if (this.previewTrack) this.applyTrack(); else this.frameMotion();
        this.status.textContent = `${this.value('recording_id') ? 'Saved take · ' : ''}${this.data.frames} frames · ${this.data.fps} fps · drag to orbit; right-drag to pan; wheel to zoom`;
    }

    resizeViewport() {
        const aspect = this.data ? this.data.width / this.data.height : 1;
        const width = Math.min(this.viewport.clientWidth, this.viewport.clientHeight * aspect);
        if (!width) return;
        this.renderer.setSize(width, width / aspect); this.camera.aspect = aspect; this.camera.updateProjectionMatrix();
    }

    updatePose() {
        if (!this.data) return;
        const f = Math.min(this.data.frames - 1, Math.floor(this.time * this.data.fps));
        const count = this.data.vertices * 3;
        this.mesh.geometry.attributes.position.array.set(this.vertices.subarray(f * count, (f + 1) * count));
        this.mesh.geometry.attributes.position.needsUpdate = true;
        this.mesh.geometry.computeVertexNormals(); this.scrub.value = this.time;
    }

    frameMotion() {
        if (!this.data) return;
        const box = new THREE.Box3(); const point = new THREE.Vector3();
        for (let i = 0; i < this.vertices.length; i += 3) box.expandByPoint(point.fromArray(this.vertices, i));
        const center = box.getCenter(new THREE.Vector3());
        const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, .1);
        const distance = radius / Math.sin(THREE.MathUtils.degToRad(20)) * Math.max(1, 1 / this.camera.aspect) * 1.05;
        const yaw = THREE.MathUtils.degToRad(this.value('yaw') ?? 0), pitch = THREE.MathUtils.degToRad(this.value('elevation') ?? 10);
        this.camera.position.copy(center).add(new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)).multiplyScalar(distance));
        this.orbit.target.copy(center); this.camera.fov = 40; this.camera.zoom = 1;
        this.camera.updateProjectionMatrix(); this.orbit.update(); this.previewTrack = false;
    }

    cameraKey(time) {
        const vector = v => ({x: v.x, y: v.y, z: v.z});
        return {time, position: vector(this.camera.position), target: vector(this.orbit.target), fov: this.camera.fov, zoom: this.camera.zoom};
    }

    capture() {
        const key = this.cameraKey(this.time);
        if (this.track.at(-1)?.time === key.time) this.track[this.track.length - 1] = key;
        else this.track.push(key);
    }

    saveTrack() {
        this.node.widgets.find(w => w.name === 'camera_track').value = JSON.stringify(this.track);
        this.node.graph?.change(); this.previewTrack = true;
    }

    finishRecording() {
        if (!this.recording) return;
        this.capture(); this.recording = false; this.saveTrack();
        this.status.textContent = `Camera saved (${this.track.length} keys). Load / render frames to apply.`;
    }

    applyTrack() {
        if (!this.track.length) return;
        let index = this.track.findIndex(k => k.time > this.time);
        const b = this.track[index < 0 ? this.track.length - 1 : index];
        const a = this.track[index <= 0 ? (index < 0 ? this.track.length - 1 : 0) : index - 1];
        const alpha = a === b ? 0 : (this.time - a.time) / (b.time - a.time);
        for (const [name, vector] of [['position', this.camera.position], ['target', this.orbit.target]]) {
            for (const axis of ['x', 'y', 'z']) vector[axis] = a[name][axis] + (b[name][axis] - a[name][axis]) * alpha;
        }
        this.camera.fov = a.fov + (b.fov - a.fov) * alpha;
        this.camera.zoom = (a.zoom ?? 1) + ((b.zoom ?? 1) - (a.zoom ?? 1)) * alpha;
        this.camera.updateProjectionMatrix(); this.orbit.update();
    }

    dispose() {
        window.removeEventListener('wheel', this.onWheel, true);
        this.disposed = true; cancelAnimationFrame(this.frame); this.resize.disconnect(); this.orbit.dispose();
        this.scene.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
        this.vertices = null; this.renderer.dispose(); this.renderer.forceContextLoss(); this.root.remove();
    }
}
