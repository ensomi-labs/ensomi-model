-- Run Ardour's bundled Vamp beat trackers on one audio file, headless, and write their raw features as JSON.
--
--   ardour9-lua ardour_vamp.lua <wav> <rate> <session-dir> <out-prefix> [bpb-list]
--
-- <session-dir> must not exist; a session at <rate> Hz is created there (Dummy backend), so that the file is read
-- at its own rate: ARDOUR.Readable.load resamples (SrcBest) only when the session's rate differs
-- (libs/ardour/readable.cc). <bpb-list> is a comma-separated list of beats-per-bar values for
-- qm-barbeattracker, default "4" (the plugin's default); qm-tempotracker runs once with its defaults.
--
-- Every run analyses channel 0 (the left channel), as Ardour's own Vamp hosts do ("analyze the first channel of
-- the audio-region", share/scripts/rubberband_swing.lua and _vamp_tempomap_example.lua), through
-- ARDOUR.LuaAPI.Vamp:analyze (libs/ardour/lua_api.cc), then getRemainingFeatures. Feature times are written as
-- the plugin reports them (Vamp::RealTime sec and nsec), with no correction.
--
-- Output: <out-prefix>__<run>.raw.json for each run (qm-bbt-bpb<k>, qm-tempo), with an "error" field when the
-- run failed. Nothing reads a chart.

local wav, rate, session_dir, out_prefix, bpb_list = arg[1], tonumber(arg[2]), arg[3], arg[4], arg[5] or "4"
assert(wav and rate and session_dir and out_prefix,
	"usage: ardour_vamp.lua <wav> <rate> <session-dir> <out-prefix> [bpb-list]")

local function q(s)
	s = tostring(s)
	s = s:gsub('[%c"\\]', function(c) return string.format("\\u%04x", c:byte()) end)
	return '"' .. s .. '"'
end

local function num(x)
	if x == nil then return "null" end
	if x ~= x or x == math.huge or x == -math.huge then return "null" end
	return string.format("%.17g", x)
end

local function write_json(path, fields)
	local fh = assert(io.open(path, "w"))
	fh:write("{\n", table.concat(fields, ",\n"), "\n}\n")
	fh:close()
end

local runs = {}
for k in string.gmatch(bpb_list, "[^,]+") do
	runs[#runs + 1] = { tag = "qm-bbt-bpb" .. k, key = "libardourvampplugins:qm-barbeattracker",
		params = { bpb = tonumber(k) }, wanted = { 0, 1 } }
end
runs[#runs + 1] = { tag = "qm-tempo", key = "libardourvampplugins:qm-tempotracker", params = {}, wanted = { 0, 2 } }

local function common(run)
	return {
		q("runner") .. ": " .. q("ardour_vamp.lua"),
		q("run") .. ": " .. q(run.tag),
		q("plugin") .. ": " .. q(run.key),
		q("file") .. ": " .. q(wav),
		q("requested_rate") .. ": " .. num(rate),
	}
end

local function fail_all(msg)
	for _, run in ipairs(runs) do
		local f = common(run)
		f[#f + 1] = q("error") .. ": " .. q(msg)
		write_json(out_prefix .. "__" .. run.tag .. ".raw.json", f)
	end
end

local s = create_session(session_dir, "vamp", rate)
if not s then
	fail_all("create_session failed")
	return
end
local sr = s:nominal_sample_rate()

local ok, rl = pcall(function() return ARDOUR.Readable.load(s, wav) end)
if not ok or rl == nil or rl:size() < 1 then
	fail_all("Readable.load failed: " .. tostring(rl))
	close_session()
	return
end
local nch = rl:size()
local rd = rl:at(0)
local len = rd:readable_length()

local function run_one(run)
	local f = common(run)
	f[#f + 1] = q("session_rate") .. ": " .. num(sr)
	f[#f + 1] = q("readable_length") .. ": " .. num(len)
	f[#f + 1] = q("channels") .. ": " .. num(nch)
	f[#f + 1] = q("channel") .. ": 0"
	local vamp = ARDOUR.LuaAPI.Vamp(run.key, sr)
	local p = vamp:plugin()
	for k, v in pairs(run.params) do p:setParameter(k, v) end
	local params = {}
	for d in p:getParameterDescriptors():iter() do
		params[#params + 1] = q(d.identifier) .. ": " .. num(p:getParameter(d.identifier))
	end
	f[#f + 1] = q("parameters") .. ": {" .. table.concat(params, ", ") .. "}"
	f[#f + 1] = q("plugin_name") .. ": " .. q(p:getName())
	f[#f + 1] = q("plugin_identifier") .. ": " .. q(p:getIdentifier())
	f[#f + 1] = q("plugin_version") .. ": " .. num(p:getPluginVersion())
	-- LuaAPI::Vamp uses the plugin's preferred sizes when both are in (0, 8192] (lua_api.cc).
	f[#f + 1] = q("step") .. ": " .. num(p:getPreferredStepSize())
	f[#f + 1] = q("block") .. ": " .. num(p:getPreferredBlockSize())
	local rv = vamp:analyze(rd, 0, nil)
	f[#f + 1] = q("analyze_rv") .. ": " .. num(rv)
	local tbl = p:getRemainingFeatures():table()
	local outs = {}
	for _, k in ipairs(run.wanted) do
		local items = {}
		local fl = tbl[k]
		if fl then
			for feat in fl:iter() do
				if feat.hasTimestamp then
					local v = "null"
					if not feat.values:empty() then v = num(feat.values:at(0)) end
					items[#items + 1] = string.format("[%d, %d, %s, %s]", feat.timestamp.sec, feat.timestamp.nsec,
						q(feat.label), v)
				end
			end
		end
		outs[#outs + 1] = q(tostring(k)) .. ": [" .. table.concat(items, ", ") .. "]"
	end
	f[#f + 1] = q("outputs") .. ": {" .. table.concat(outs, ",\n") .. "}"
	vamp:reset()
	write_json(out_prefix .. "__" .. run.tag .. ".raw.json", f)
end

for _, run in ipairs(runs) do
	local good, err = pcall(run_one, run)
	if not good then
		local f = common(run)
		f[#f + 1] = q("session_rate") .. ": " .. num(sr)
		f[#f + 1] = q("error") .. ": " .. q(err)
		write_json(out_prefix .. "__" .. run.tag .. ".raw.json", f)
	end
end

close_session()
