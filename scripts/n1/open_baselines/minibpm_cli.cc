// Print Ardour's MiniBPM estimates for one WAV file as JSON, before Ardour rounds them.
//
// Build against Ardour 9.8's own libs/ardour/minibpm.cc and libs/ardour/ardour/minibpm.h (GPL-2.0-or-later,
// not in this repository), with the optimisation flags Ardour's wscript uses for clang on macOS:
//
//   clang++ -std=c++11 -O3 -ffast-math -fno-finite-math-only -I <ardour>/libs/ardour \
//       minibpm_cli.cc <ardour>/libs/ardour/minibpm.cc -o minibpm_cli
//   minibpm_cli <file.wav>
//
// Ardour hands MiniBPM float samples of one channel, channel 0 for a clip (AudioTrigger::estimate_tempo,
// audio_data (0)), with MiniBPM's defaults (55-190 BPM, 4 beats per bar). Two estimates are printed:
//   "whole":    the whole file, as a clip trigger estimates a dropped file (libs/ardour/triggerbox.cc);
//   "middle10": ten seconds from the middle, as AudioSource::estimate_tempo does for recordings and embedded
//               files (libs/ardour/audiosource.cc); null below ten seconds.
// Reads 32-bit float (WAVE_FORMAT_IEEE_FLOAT or WAVE_FORMAT_EXTENSIBLE with the float subformat) and 16-bit
// PCM. The rounding Ardour applies afterwards is in obtools.py (ardour_rounded_tempo), not here.

#include "ardour/minibpm.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static uint16_t u16(const unsigned char* p) { return uint16_t(p[0] | (p[1] << 8)); }
static uint32_t u32(const unsigned char* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }

static int fail(const std::string& msg)
{
	std::printf("{\"error\": \"%s\"}\n", msg.c_str());
	return 2;
}

static void print_estimate(const char* name, double bpm, const std::vector<double>& cands, bool comma)
{
	std::printf("  \"%s\": {\"bpm\": %.17g, \"candidates\": [", name, bpm);
	for (size_t i = 0; i < cands.size(); ++i) {
		std::printf("%s%.17g", i ? ", " : "", cands[i]);
	}
	std::printf("]}%s\n", comma ? "," : "");
}

int main(int argc, char** argv)
{
	if (argc != 2) {
		std::fprintf(stderr, "usage: minibpm_cli <file.wav>\n");
		return 1;
	}
	std::ifstream in(argv[1], std::ios::binary);
	if (!in) {
		return fail("cannot open file");
	}
	std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (b.size() < 12 || std::memcmp(&b[0], "RIFF", 4) || std::memcmp(&b[8], "WAVE", 4)) {
		return fail("not a RIFF/WAVE file");
	}
	unsigned tag = 0, channels = 0, rate = 0, align = 0, bits = 0;
	size_t data_off = 0, data_len = 0;
	for (size_t pos = 12; pos + 8 <= b.size();) {
		uint32_t size = u32(&b[pos + 4]);
		if (!std::memcmp(&b[pos], "fmt ", 4) && pos + 8 + 16 <= b.size()) {
			const unsigned char* f = &b[pos + 8];
			tag = u16(f);
			channels = u16(f + 2);
			rate = u32(f + 4);
			align = u16(f + 12);
			bits = u16(f + 14);
			if (tag == 0xFFFE && size >= 26 && pos + 8 + 26 <= b.size()) {
				tag = u16(f + 24);
			}
		} else if (!std::memcmp(&b[pos], "data", 4)) {
			data_off = pos + 8;
			data_len = std::min<size_t>(size, b.size() - data_off);
		}
		pos += 8 + size + (size & 1);
	}
	if (!data_off || !channels || !rate || !align) {
		return fail("missing fmt or data chunk");
	}
	const size_t frames = data_len / align;
	std::vector<float> x(frames);
	if (tag == 3 && bits == 32) {
		for (size_t i = 0; i < frames; ++i) {
			std::memcpy(&x[i], &b[data_off + i * align], 4);
		}
	} else if (tag == 1 && bits == 16) {
		for (size_t i = 0; i < frames; ++i) {
			int16_t v;
			std::memcpy(&v, &b[data_off + i * align], 2);
			x[i] = v / 32768.0f;
		}
	} else {
		return fail("unsupported sample format");
	}

	breakfastquay::MiniBPM whole(rate);
	const double bpm = whole.estimateTempoOfSamples(&x[0], (int)frames);
	const std::vector<double> cands = whole.getTempoCandidates();

	std::printf("{\n  \"runner\": \"minibpm_cli\",\n  \"sample_rate\": %u,\n  \"channels\": %u,\n  \"format_tag\": %u,\n"
	            "  \"frames\": %zu,\n  \"channel_used\": 0,\n", rate, channels, tag, frames);
	const size_t ten = (size_t)rate * 10;
	if (frames >= ten) {
		print_estimate("whole", bpm, cands, true);
		const size_t start = (frames - ten) / 2;
		breakfastquay::MiniBPM middle(rate);
		const double mbpm = middle.estimateTempoOfSamples(&x[start], (int)ten);
		std::printf("  \"middle10_start\": %zu,\n", start);
		print_estimate("middle10", mbpm, middle.getTempoCandidates(), false);
	} else {
		print_estimate("whole", bpm, cands, true);
		std::printf("  \"middle10\": null\n");
	}
	std::printf("}\n");
	return 0;
}
