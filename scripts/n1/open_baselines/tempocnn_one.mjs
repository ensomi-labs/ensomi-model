// Run openDAW's own TempoDetectionTask (TempoCNN) headless in Node on one WAV file and write its result as JSON.
//
//   node --import <H>/node_resolve.mjs <H>/tempocnn_one.mjs <opendaw> <model.onnx> <file.wav> <out.json>
//
// <H> is a directory holding this file and node_resolve.mjs, whose node_modules is a symlink to the openDAW
// checkout's node_modules, so bare imports resolve to the checkout's packages (the trial harness of 2026-10-21,
// research notes open-daws-20261021/opendaw/od_detect_headless.mjs). <opendaw> must stay the first argument:
// node_resolve.mjs reads it.
//
// What openDAW does, and what this does:
// - "Detect BPM (AI)..." passes the region's file data, left channel and native rate, to the task:
//   TempoDetection.runOne(data.frames[0], data.sampleRate, ...) (RegionContextMenu.ts lines 194-206). Here the
//   task gets channel 0 of the canonical decode at its own rate.
// - openDAW decodes a WAV with its own WavFile.decodeFloats and falls back to the browser's decodeAudioData,
//   which can resample (SampleService.ts). This file reads the WAV itself, then also tries WavFile.decodeFloats
//   and records whether openDAW's decoder accepts the file and gives the same left channel.
// - The task's front end (resampling to 11025 Hz, mel, windows, octave rule) is openDAW's code, run through
//   onnxruntime-web with the wasm execution provider, as in openDAW. Nothing here changes openDAW's code.

import crypto from "node:crypto"
import fs from "node:fs"
import path from "node:path"
import {pathToFileURL} from "node:url"

const [repo, modelPath, wavPath, outPath] = process.argv.slice(2)
if (!repo || !modelPath || !wavPath || !outPath) {
    console.error("usage: node tempocnn_one.mjs <opendaw> <model.onnx> <file.wav> <out.json>")
    process.exit(1)
}

const parseWav = buffer => {
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)
    const ascii = (o, n) => buffer.toString("ascii", o, o + n)
    if (ascii(0, 4) !== "RIFF" || ascii(8, 4) !== "WAVE") throw new Error("not a RIFF/WAVE file")
    let fmt = null, dataOffset = -1, dataBytes = 0
    for (let o = 12; o + 8 <= buffer.length;) {
        const id = ascii(o, 4), size = view.getUint32(o + 4, true)
        if (id === "fmt ") {
            let tag = view.getUint16(o + 8, true)
            if (tag === 0xFFFE && size >= 26) tag = view.getUint16(o + 8 + 24, true)
            fmt = {formatTag: view.getUint16(o + 8, true), tag, channels: view.getUint16(o + 10, true),
                rate: view.getUint32(o + 12, true), align: view.getUint16(o + 20, true), bits: view.getUint16(o + 22, true)}
        }
        if (id === "data") {
            dataOffset = o + 8
            dataBytes = Math.min(size, buffer.length - dataOffset)
        }
        o += 8 + size + (size & 1)
    }
    if (!fmt || dataOffset < 0) throw new Error("missing fmt or data chunk")
    const frames = Math.floor(dataBytes / fmt.align)
    const left = new Float32Array(frames)
    if (fmt.tag === 3 && fmt.bits === 32) {
        for (let i = 0; i < frames; i++) left[i] = view.getFloat32(dataOffset + i * fmt.align, true)
    } else if (fmt.tag === 1 && fmt.bits === 16) {
        for (let i = 0; i < frames; i++) left[i] = view.getInt16(dataOffset + i * fmt.align, true) / 32768
    } else {
        throw new Error(`unsupported WAV format ${fmt.tag} at ${fmt.bits} bits`)
    }
    return {...fmt, frames, left}
}

const result = {runner: "tempocnn_one.mjs", wav: wavPath, node: process.version, channel_used: 0}
try {
    const fileBuffer = fs.readFileSync(wavPath)
    const wav = parseWav(fileBuffer)
    Object.assign(result, {sampleRate: wav.rate, frames: wav.frames, channels: wav.channels, format_tag: wav.formatTag})
    try {
        const {WavFile} = await import("@opendaw/lib-dsp")
        const arrayBuffer = fileBuffer.buffer.slice(fileBuffer.byteOffset, fileBuffer.byteOffset + fileBuffer.byteLength)
        const data = WavFile.decodeFloats(arrayBuffer)
        const same = data.sampleRate === wav.rate && data.frames[0].length === wav.left.length
            && data.frames[0].every((v, i) => Object.is(v, wav.left[i]))
        result.opendaw_wav_decoder = "accepted"
        result.opendaw_left_equal = same
    } catch (error) {
        result.opendaw_wav_decoder = `rejected: ${error}`
        result.opendaw_left_equal = null
    }
    const modelBytes = fs.readFileSync(modelPath)
    result.model_sha256 = crypto.createHash("sha256").update(modelBytes).digest("hex")
    const ort = await import("onnxruntime-web")
    const {Option} = await import("@opendaw/lib-std")
    const taskFile = path.join(repo, "packages/lib/inference/dist/tasks/TempoDetectionTask.js")
    const {TempoDetectionTask} = await import(pathToFileURL(taskFile).href)
    result.onnxruntime = ort.env?.versions ?? null
    const session = await ort.InferenceSession.create(modelBytes, {executionProviders: ["wasm"]})
    const env = {
        session: async feeds => {
            const input = {}
            for (const [name, t] of Object.entries(feeds)) input[name] = new ort.Tensor(t.type, t.data, t.dims)
            const output = await session.run(input)
            const out = {}
            for (const [name, t] of Object.entries(output)) out[name] = {type: t.type, data: t.data, dims: t.dims}
            return out
        },
        progress: () => {},
        signal: Option.None,
        inputNames: session.inputNames,
        outputNames: session.outputNames
    }
    const started = Date.now()
    const out = await TempoDetectionTask.run({audio: wav.left, sampleRate: wav.rate}, env)
    Object.assign(result, out, {ms: Date.now() - started})
    console.log(`${path.basename(wavPath)}: ${out.bpm} bpm, confidence ${out.confidence.toFixed(3)}`)
} catch (error) {
    result.error = String(error && error.stack || error)
    console.log(`${path.basename(wavPath)}: ERROR ${error}`)
}
fs.writeFileSync(outPath, JSON.stringify(result, null, 1))
