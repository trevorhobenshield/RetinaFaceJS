let T, session, score, bbox, kps;

const MODEL_PATH = 'det_10g.onnx';
const MEAN = 127.5;
const STD = 128.0;
const INPUT_HEIGHT = 640;
const INPUT_WIDTH = 640;
const INPUT_DIMS = [1, 3, INPUT_HEIGHT, INPUT_WIDTH];
const FEAT_STRIDE_FPN = [8, 16, 32];
const FMC = 3;
const NUM_ANCHORS = 2;
const THRESHOLD = 0.5;
const USE_KPS = true;
const CENTER_CACHE = {};


function reshape(arr, m, n) {
    let A = [];
    for (let i = 0; i < m; i++) {
        let row = [];
        for (let j = 0; j < n; j++) {
            row.push(arr[i * n + j]);
        }
        A.push(row);
    }
    return A;
}

function generate_anchor_centers(height, width) {
    return Array.from({length: height}, (_, y) => Array.from({length: width}, (_, x) => [x, y]));
}

function scale_reshape_ac(anchor_centers, stride) {
    let res = [];
    for (let y = 0; y < anchor_centers.length; y++) {
        for (let x = 0; x < anchor_centers[0].length; x++) {
            let scaledX = anchor_centers[y][x][0] * stride;
            let scaledY = anchor_centers[y][x][1] * stride;
            res.push([scaledX, scaledY]);
        }
    }
    return res;
}

function stack_reshape_ac(anchor_centers, num_anchors) {
    let stacked = [];
    for (let i = 0; i < anchor_centers.length; i++) {
        for (let j = 0; j < num_anchors; j++) {
            stacked.push(anchor_centers[i]);
        }
    }
    return stacked;
}

function distance2bbox(anchor_centers, bbox_preds, max_shape = null) {
    return anchor_centers.map((center, i) => {
        let [x_center, y_center] = center;
        let [dx1, dy1, dx2, dy2] = bbox_preds[i];
        let x1 = Math.max(0, Math.min(x_center - dx1, max_shape ? max_shape[1] : Infinity));
        let y1 = Math.max(0, Math.min(y_center - dy1, max_shape ? max_shape[0] : Infinity));
        let x2 = Math.max(0, Math.min(x_center + dx2, max_shape ? max_shape[1] : Infinity));
        let y2 = Math.max(0, Math.min(y_center + dy2, max_shape ? max_shape[0] : Infinity));
        return [x1, y1, x2, y2];
    });
}

function distance2kps(points, distance, max_shape = null) {
    return points.map((point, i) =>
        distance[i].reduce((acc, d, j) => {
            let p = point[j % 2] + d;
            p = max_shape ? Math.min(Math.max(p, 0), max_shape[j % 2 === 0 ? 1 : 0]) : p;
            return [...acc, p];
        }, []),
    );
}

function reshape_kpss(kpss) {
    const reshaped = [];
    kpss.forEach(row => {
        const newRow = [];
        for (let i = 0; i < row.length; i += 2) {
            newRow.push([row[i], row[i + 1]]);
        }
        reshaped.push(newRow);
    });
    return reshaped;
}

function nms(dets, threshold) {
    const x1 = dets.map(det => det[0]);
    const y1 = dets.map(det => det[1]);
    const x2 = dets.map(det => det[2]);
    const y2 = dets.map(det => det[3]);
    const scores = dets.map(det => det[4]);

    const areas = x1.map((x, i) => (x2[i] - x + 1) * (y2[i] - y1[i] + 1));
    let order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);

    let keep = [];
    while (order.length > 0) {
        const i = order.shift();
        keep.push(i);

        const ovr = order.map(o => {
            const xx1 = Math.max(x1[i], x1[o]);
            const yy1 = Math.max(y1[i], y1[o]);
            const xx2 = Math.min(x2[i], x2[o]);
            const yy2 = Math.min(y2[i], y2[o]);

            const w = Math.max(0, xx2 - xx1 + 1);
            const h = Math.max(0, yy2 - yy1 + 1);
            const inter = w * h;

            return inter / (areas[i] + areas[o] - inter);
        });
        order = order.filter((_, i) => ovr[i] <= threshold);
    }
    return keep;
}

function get_det_kpss(det_scale, scores_list, bboxes_list, kpss_list) {
    let scores = scores_list.flat(2);
    const order = Array.from({length: scores.length}, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
    let bboxes = bboxes_list.flat().map(box => box.map(val => val / det_scale));
    let kpss = kpss_list.flat().map(kpsGroup => kpsGroup.map(kps => kps.map(val => val / det_scale)));
    let pre_det = bboxes.map((box, i) => [...box, scores[i]]);
    pre_det = order.map(o => pre_det[o]);

    let keep = nms(pre_det, THRESHOLD);
    let det = keep.map(k => pre_det[k]);
    let selected_kpss = [];
    if (kpss.length > 0) {
        selected_kpss = keep.map(k => kpss[Math.floor(order[k] / kpss[0].length)]).filter(k => k);
    }
    return [det, selected_kpss];
}

function forward(out) {
    let scores_list = [];
    let bboxes_list = [];
    let kpss_list = [];
    for (let [i, stride] of FEAT_STRIDE_FPN.entries()) {
        let scores = out[i].data;
        const tmp = out[i + FMC];
        let bbox_preds = reshape(tmp.data, ...tmp.dims);
        bbox_preds = bbox_preds.map(arr => arr.map(x => x * stride));

        let kps_preds;
        if (USE_KPS) {
            let tmp = out[i + FMC * 2];
            tmp = reshape(tmp.data, ...tmp.dims);
            kps_preds = tmp.map(arr => arr.map(x => x * stride));
        }

        let height = Math.floor(INPUT_HEIGHT / stride);
        let width = Math.floor(INPUT_WIDTH / stride);
        let key = String([height, width, stride]); // can't use array as key in js

        let anchor_centers;
        if (CENTER_CACHE[key]) {
            anchor_centers = CENTER_CACHE[key];
        } else {
            anchor_centers = generate_anchor_centers(height, width);
            anchor_centers = scale_reshape_ac(anchor_centers, stride);

            if (NUM_ANCHORS > 1) {
                anchor_centers = stack_reshape_ac(anchor_centers, NUM_ANCHORS);
            }
            if (Object.keys(CENTER_CACHE).length < 100) {
                CENTER_CACHE[key] = anchor_centers;
            }
        }

        let pos_inds = [...scores.map((e, i) => (e >= THRESHOLD ? i : 0)).filter(x => x > 0)];
        let bboxes = distance2bbox(anchor_centers, bbox_preds);
        let pos_scores = pos_inds.map(x => scores[x]);
        let pos_bboxes = pos_inds.map(x => bboxes[x]);
        scores_list.push(pos_scores);
        bboxes_list.push(pos_bboxes);

        let kpss, pos_kpss;
        if (USE_KPS) {
            kpss = distance2kps(anchor_centers, kps_preds);
            kpss = reshape_kpss(kpss);
            pos_kpss = pos_inds.map(x => kpss[x]);
            kpss_list.push(pos_kpss);
        }
    }
    return [scores_list, bboxes_list, kpss_list];
}

function pad_image(source, sourceW, sourceH, channels = 4, targetW = 640, targetH = 640) {
    const A = new Float32Array(targetW * targetH * channels);
    for (let y = 0; y < sourceH; y++) {
        for (let x = 0; x < sourceW; x++) {
            const i = (y * sourceW + x) * channels;
            const j = (y * targetW + x) * channels;
            for (let k = 0; k < channels; k++) {
                A[j + k] = source[i + k];
            }
        }
    }
    return A;
}

function strip_alpha(arr, w, h) {
    const A = new Float32Array(w * h * 3);
    for (let i = 0, j = 0; i < arr.length; i += 4, j += 3) {
        A[j] = arr[i];
        A[j + 1] = arr[i + 1];
        A[j + 2] = arr[i + 2];
    }
    return A;
}

function resize_image(imgElement, w, h) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const aspectRatio = imgElement.width / imgElement.height;
    let newWidth = w;
    let newHeight = Math.round(newWidth / aspectRatio);
    if (newHeight > h) {
        newHeight = h;
        newWidth = Math.round(newHeight * aspectRatio);
    }
    canvas.width = newWidth;
    canvas.height = newHeight;
    ctx.drawImage(imgElement, 0, 0, newWidth, newHeight);
    return ctx.getImageData(0, 0, newWidth, newHeight);
}

function img_to_tensor(imgElement, dims, mapfn) {
    const pad_dims = [dims[1] + 1, dims[2], dims[3]]; // E.g. [4, 640, 640];
    const {data, width, height} = resize_image(imgElement, ...pad_dims.slice(1));
    const normed = Float32Array.from(data, mapfn); // norm before pad
    const padded = pad_image(normed, width, height, ...pad_dims);
    const stripped = strip_alpha(padded, ...pad_dims.slice(1));
    const RGB = Float32Array.from(transpose_rgb(stripped));
    return new ort.Tensor('float32', RGB, dims);
}

function transpose_rgb(arr) {
    const [R, G, B] = [[], [], []];
    for (let i = 0; i < arr.length; i += 3) {
        R.push(arr[i]);
        G.push(arr[i + 1]);
        B.push(arr[i + 2]);
    }
    return [...R, ...G, ...B];
}

async function main() {
    const start = performance.now();
    try {
        session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            executionMode: 'parallel',
            enableCpuMemArena: true,
            enableMemPattern: true,
            extra: {
                optimization: {
                    enable_gelu_approximation: '1',
                },
            },
        });
        const img = document.querySelector('img');
        const det_scale = INPUT_HEIGHT / img.height;
        T = img_to_tensor(img, INPUT_DIMS, x => (x - MEAN) / STD);
        const feeds = {[session.inputNames[0]]: T}
        console.log(`%c[${T.type}]%c(${T.dims}) => ${MODEL_PATH}`, 'color:green', null);
        let out = await session.run(feeds);
        out = Object.values(out).sort((a, b) => a.dims.slice(-1) - b.dims.slice(-1));

        const [scores_list, bboxes_list, kpss_list] = forward(out);
        const [det, kpss] = get_det_kpss(det_scale, scores_list, bboxes_list, kpss_list);
        score = det.flat().pop()
        bbox = det.flat().slice(0, 4).map(x => Math.ceil(x))
        kps = kpss.flat()

        console.log('score =', score)
        console.log('bbox =', bbox)
        console.log('kps =', kps)

    } catch (e) {
        console.log(e);
        document.body.style.backgroundColor = 'black';
        document.body.style.color = 'green';
        document.body.innerHTML = `<code>${e}</code>`;
    }
    const end = performance.now();
    console.log(`took ${(end - start) / 1000}s`);
}
