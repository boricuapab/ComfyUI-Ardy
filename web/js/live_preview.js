import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';
import { api } from '../../../scripts/api.js';
import { queuePreview } from './queue.js';

export class ArdyViewer {
    constructor(node) {
        this.node = node;
        this.active = false;
        this.paused = false;
        this.played = 0;
        this.chunks = [];
        this.keys = new Set();
        this.waypoints = [];
        this.root = document.createElement('div');
        this.root.style.cssText = 'height:540px;width:100%;display:flex;flex-direction:column;background:#171d25;color:#e5edf5;font:12px sans-serif;border-radius:8px;overflow:hidden;';
        const bar = document.createElement('div');
        bar.style.cssText = 'display:flex;gap:8px;align-items:center;padding:8px;flex-wrap:wrap';
        this.root.append(bar);
        const button = (label, action) => {
            const b = document.createElement('button');
            b.textContent = label;
            b.style.cssText = 'background:#34485d;color:white;border:0;border-radius:4px;padding:7px 10px;cursor:pointer';
            b.onclick = action;
            bar.append(b);
            return b;
        };
        this.start = button('Start new take', async () => {
            if (this.active) return;
            this.node.widgets.find(w => w.name === 'control_id').value = crypto.randomUUID();
            this.status.textContent = 'Queued — loading ARDY…';
            try { await queuePreview(this.node); }
            catch (error) { this.status.textContent = error.message; }
        });
        this.pause = button('Pause', () => {
            this.paused = !this.paused;
            this.pause.textContent = this.paused ? 'Resume' : 'Pause';
            this.send({paused: this.paused});
        });
        button('Stop & save take', () => this.send({stop: true, played: this.played}));
        button('Clear route', () => {
            this.marker.visible = false;
            this.setWaypoints([]);
        });
        button('Frame character', () => this.frameCharacter());
        this.mode = document.createElement('select');
        for (const label of ['Mesh', 'Mesh + skeleton', 'Skeleton']) {
            const option = document.createElement('option'); option.textContent = label; this.mode.append(option);
        }
        this.mode.onchange = () => {
            if (this.mesh) this.mesh.visible = this.mode.value !== 'Skeleton';
            if (this.bones) this.bones.visible = this.mode.value !== 'Mesh';
        };
        bar.append(this.mode);
        const followLabel = document.createElement('label');
        this.follow = document.createElement('input'); this.follow.type = 'checkbox'; this.follow.checked = true;
        followLabel.append(this.follow, ' Follow'); bar.append(followLabel);
        this.viewport = document.createElement('div');
        this.viewport.style.cssText = 'flex:1;min-height:260px;position:relative';
        this.root.append(this.viewport);
        this.status = document.createElement('div');
        this.status.style.cssText = 'padding:8px;min-height:16px';
        this.status.textContent = 'Start new take to record. Click to add waypoints; drag to orbit; right-drag to pan.';
        this.root.append(this.status);
        this.scene = new THREE.Scene(); this.scene.background = new THREE.Color('#171d25');
        this.camera = new THREE.PerspectiveCamera(45, 1, .01, 1000);
        this.camera.position.set(5, 5, 7);
        this.renderer = new THREE.WebGLRenderer({antialias: true});
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.viewport.append(this.renderer.domElement);
        this.renderer.domElement.tabIndex = 0;
        this.renderer.domElement.style.outline = 'none';
        this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
        this.orbit.target.set(0, 1, 0); this.orbit.update();
        this.orbit.mouseButtons.RIGHT = THREE.MOUSE.PAN;
        this.scene.add(new THREE.HemisphereLight(0xddefff, 0x48535b, 2.2));
        const light = new THREE.DirectionalLight(0xffffff, 2.4); light.position.set(3, 8, 5); this.scene.add(light);
        this.grid = new THREE.GridHelper(200, 200, 0x697d91, 0x354452); this.scene.add(this.grid);
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({color: 0x202b35, roughness: 1}));
        floor.rotation.x = -Math.PI / 2; floor.position.y = -.015; this.scene.add(floor);
        this.marker = new THREE.Mesh(new THREE.RingGeometry(.16, .24, 32), new THREE.MeshBasicMaterial({color: 0xffb84c, side: THREE.DoubleSide}));
        this.marker.rotation.x = -Math.PI / 2; this.marker.visible = false; this.scene.add(this.marker);
        this.route = new THREE.Group(); this.scene.add(this.route);
        const ray = new THREE.Raycaster();
        const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
        let down = null;
        this.renderer.domElement.addEventListener('pointerdown', e => {
            this.renderer.domElement.focus({preventScroll: true});
            if (e.button === 0) down = [e.clientX, e.clientY];
        });
        this.renderer.domElement.addEventListener('pointerup', e => {
            if (!down || e.button !== 0) return;
            const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]); down = null;
            if (moved > 5) return;
            const rect = this.renderer.domElement.getBoundingClientRect();
            ray.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1,
                -(e.clientY - rect.top) / rect.height * 2 + 1), this.camera);
            const point = ray.ray.intersectPlane(plane, new THREE.Vector3());
            if (!point) return;
            this.marker.position.set(point.x, .02, point.z); this.marker.visible = true;
            this.setWaypoints([...this.waypoints, [point.x, point.z]]);
            this.status.textContent = `Target: ${point.x.toFixed(1)}, ${point.z.toFixed(1)} — steering next chunk`;
        });
        this.root.addEventListener('pointerdown', e => e.stopPropagation());
        this.root.addEventListener('pointerup', e => e.stopPropagation());
        this.root.addEventListener('contextmenu', e => { e.preventDefault(); e.stopPropagation(); });
        this.root.addEventListener('wheel', e => e.stopPropagation());
        this.renderer.domElement.addEventListener('keydown', e => {
            if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return;
            e.preventDefault(); e.stopPropagation(); this.keys.add(e.key);
            this.send({direction: this.keyboardDirection()});
        });
        this.renderer.domElement.addEventListener('keyup', e => {
            if (!this.keys.has(e.key)) return;
            e.preventDefault(); e.stopPropagation(); this.keys.delete(e.key);
            this.send({direction: this.keyboardDirection()});
        });
        this.clearKeys = () => {
            this.keys.clear(); this.send({direction: null});
        };
        this.renderer.domElement.addEventListener('blur', this.clearKeys);
        window.addEventListener('blur', this.clearKeys);
        this.resize = new ResizeObserver(() => {
            const w = this.viewport.clientWidth, h = this.viewport.clientHeight;
            if (!w || !h) return;
            this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
        });
        this.resize.observe(this.viewport);
        this.listener = e => {
            if (e.detail.control_id === this.value('control_id')) this.receive(e.detail);
        };
        api.addEventListener('ardy.live', this.listener);
        this.heartbeat = setInterval(() => {
            const value = this.value('waypoints') ?? '[]';
            if (value !== this.lastWaypoints) {
                try { this.setWaypoints(JSON.parse(value.trim() || '[]')); }
                catch { this.status.textContent = 'Waypoints must be a JSON list of [x, z] coordinates.'; }
            }
            if (this.active) this.send({played: this.played, prompt: this.value('prompt'), speed: this.value('speed'), direction: this.keyboardDirection()});
        }, 500);
        this.last = performance.now();
        const animate = now => {
            this.frame = requestAnimationFrame(animate);
            const dt = Math.min((now - this.last) / 1000, .1); this.last = now;
            if (this.active && !this.paused && this.chunks.length) {
                const end = this.chunks.at(-1).start + this.chunks.at(-1).positions.length - 1;
                this.played = Math.min(this.played + dt * this.fps, end);
                while (this.chunks.length > 1 && this.played >= this.chunks[1].start) this.chunks.shift();
                const chunk = this.chunks[0];
                const f = Math.max(0, Math.min(Math.floor(this.played - chunk.start), chunk.positions.length - 1));
                const next = Math.min(f + 1, chunk.positions.length - 1);
                this.pose(chunk.positions[f], chunk.rotations[f], chunk.positions[next], chunk.rotations[next], this.played % 1);
            }
            this.renderer.render(this.scene, this.camera);
        };
        this.frame = requestAnimationFrame(animate);
    }

    value(name) { return this.node.widgets?.find(w => w.name === name)?.value; }

    keyboardDirection() {
        const forward = new THREE.Vector3().subVectors(this.orbit.target, this.camera.position);
        forward.y = 0; forward.normalize();
        const right = new THREE.Vector3(-forward.z, 0, forward.x);
        const direction = forward.multiplyScalar(Number(this.keys.has('ArrowUp')) - Number(this.keys.has('ArrowDown')))
            .addScaledVector(right, Number(this.keys.has('ArrowRight')) - Number(this.keys.has('ArrowLeft')));
        return direction.lengthSq() ? [direction.x, direction.z] : null;
    }

    setWaypoints(points) {
        if (!Array.isArray(points) || points.some(p => !Array.isArray(p) || p.length !== 2 || p.some(v => typeof v !== 'number' || !Number.isFinite(v))))
            throw new Error('Invalid waypoints');
        const append = points.length > this.waypoints.length && this.waypoints.every((p, i) => p[0] === points[i][0] && p[1] === points[i][1]);
        const added = points.slice(this.waypoints.length);
        this.waypoints = points;
        this.lastWaypoints = JSON.stringify(points);
        const widget = this.node.widgets.find(w => w.name === 'waypoints');
        if (widget) widget.value = this.lastWaypoints;
        for (const child of [...this.route.children]) {
            child.geometry.dispose(); child.material.dispose(); this.route.remove(child);
        }
        for (const point of points) {
            const ring = new THREE.Mesh(new THREE.RingGeometry(.12, .19, 24), new THREE.MeshBasicMaterial({color: 0xffb84c, side: THREE.DoubleSide}));
            ring.rotation.x = -Math.PI / 2; ring.position.set(point[0], .025, point[1]); this.route.add(ring);
        }
        if (points.length > 1) {
            const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], .025, p[1])));
            this.route.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({color: 0xffb84c})));
        }
        this.send(append ? {append_waypoints: added} : {waypoints: points});
    }

    async send(data) {
        if (!this.active) return;
        try {
            const response = await api.fetchApi(`/ardy/live/${encodeURIComponent(this.value('control_id'))}`, {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
            });
            if (!response.ok) {
                const error = await response.json();
                this.status.textContent = error.error;
                if (response.status === 404) this.active = false;
            }
        } catch (error) { this.status.textContent = `Connection lost: ${error.message}`; }
    }

    receive(data) {
        if (data.kind === 'start') {
            this.active = true; this.paused = false; this.pause.textContent = 'Pause';
            this.played = 0; this.chunks = []; this.fps = data.fps;
            this.buildBody(data.body, data.parents);
            this.start.disabled = true;
            this.send({ready: true, played: 0});
            this.status.textContent = 'Loading motion…';
        } else if (data.kind === 'chunk') {
            this.chunks.push(data);
            if (data.start === 0) {
                this.pose(data.positions[0], data.rotations[0], data.positions[0], data.rotations[0], 0);
                this.frameCharacter();
            }
            this.status.textContent = `Recording · ${(this.played / this.fps).toFixed(1)}s · Stop & save take to finish`;
        } else if (data.kind === 'status') this.status.textContent = data.message;
        else if (data.kind === 'stop') {
            this.active = false; this.start.disabled = false;
            this.status.textContent = 'Take saved. Run reuses it; Start new take records again.';
        }
    }

    buildBody(body, parents) {
        for (const object of [this.mesh, this.bones]) {
            if (object) { this.scene.remove(object); object.geometry.dispose(); object.material.dispose(); }
        }
        this.bind = new Float32Array(body.bind_vertices.flat());
        this.indices = body.lbs_indices;
        this.weights = body.lbs_weights;
        this.parents = parents;
        this.inverseBind = body.bind_rig_transform.map(m => new THREE.Matrix4().set(...m.flat()).invert());
        this.transforms = parents.map(() => new THREE.Matrix4());
        this.positions = parents.map(() => new THREE.Vector3());
        this.rotations = parents.map(() => new THREE.Quaternion());
        this.nextRotation = new THREE.Quaternion();
        this.scale = new THREE.Vector3(1, 1, 1);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(this.bind.slice(), 3).setUsage(THREE.DynamicDrawUsage));
        geometry.setIndex(body.faces.flat()); geometry.computeVertexNormals();
        this.mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: 0x81c9ec, roughness: .65, side: THREE.DoubleSide}));
        this.mesh.frustumCulled = false; this.scene.add(this.mesh);
        const lines = new THREE.BufferGeometry();
        lines.setAttribute('position', new THREE.BufferAttribute(new Float32Array(parents.filter(p => p >= 0).length * 6), 3));
        this.bones = new THREE.LineSegments(lines, new THREE.LineBasicMaterial({color: 0xffce70, depthTest: false}));
        this.bones.frustumCulled = false; this.bones.renderOrder = 2; this.scene.add(this.bones);
        this.mode.onchange(); this.lastRoot = null;
    }

    pose(p, q, p2, q2, alpha) {
        if (!this.mesh) return;
        for (let j = 0; j < p.length; j++) {
            this.positions[j].set(p[j][0] + (p2[j][0] - p[j][0]) * alpha,
                p[j][1] + (p2[j][1] - p[j][1]) * alpha, p[j][2] + (p2[j][2] - p[j][2]) * alpha);
            this.rotations[j].fromArray(q[j]).slerp(this.nextRotation.fromArray(q2[j]), alpha);
            this.transforms[j].compose(this.positions[j], this.rotations[j], this.scale).multiply(this.inverseBind[j]);
        }
        // Match CoreSkin.lbs, including the fifth influence supported by the ARDY asset.
        const output = this.mesh.geometry.attributes.position.array;
        for (let v = 0; v < this.indices.length; v++) {
            const x = this.bind[v * 3], y = this.bind[v * 3 + 1], z = this.bind[v * 3 + 2];
            let ox = 0, oy = 0, oz = 0;
            for (let k = 0; k < this.indices[v].length; k++) {
                const w = this.weights[v][k], m = this.transforms[this.indices[v][k]].elements;
                ox += w * (m[0] * x + m[4] * y + m[8] * z + m[12]);
                oy += w * (m[1] * x + m[5] * y + m[9] * z + m[13]);
                oz += w * (m[2] * x + m[6] * y + m[10] * z + m[14]);
            }
            output[v * 3] = ox; output[v * 3 + 1] = oy; output[v * 3 + 2] = oz;
        }
        this.mesh.geometry.attributes.position.needsUpdate = true;
        this.mesh.geometry.computeVertexNormals();
        const lines = this.bones.geometry.attributes.position.array;
        let offset = 0;
        this.parents.forEach((parent, j) => {
            if (parent < 0) return;
            this.positions[parent].toArray(lines, offset); this.positions[j].toArray(lines, offset + 3); offset += 6;
        });
        this.bones.geometry.attributes.position.needsUpdate = true;
        const root = this.positions[0];
        if (this.follow.checked && this.lastRoot) {
            const dx = root.x - this.lastRoot.x, dz = root.z - this.lastRoot.z;
            this.camera.position.x += dx; this.camera.position.z += dz;
            this.orbit.target.x += dx; this.orbit.target.z += dz; this.orbit.update();
        }
        this.lastRoot = root.clone();
    }

    frameCharacter() {
        const root = this.positions?.[0] ?? new THREE.Vector3();
        this.orbit.target.set(root.x, root.y + .1, root.z);
        this.camera.position.set(root.x + 4, root.y + 3, root.z + 5); this.orbit.update();
    }

    dispose() {
        window.removeEventListener('blur', this.clearKeys);
        this.send({stop: true}); this.active = false;
        cancelAnimationFrame(this.frame); clearInterval(this.heartbeat);
        this.resize.disconnect(); this.orbit.dispose(); api.removeEventListener('ardy.live', this.listener);
        this.scene.traverse(o => { o.geometry?.dispose(); if (o.material) o.material.dispose(); });
        this.renderer.dispose(); this.renderer.forceContextLoss(); this.root.remove();
    }
}
