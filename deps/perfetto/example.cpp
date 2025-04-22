#include <perfetto.h>

// The set of track event categories that the example is using.
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("rendering")
        .SetDescription("Rendering and graphics events"),
    perfetto::Category("network.debug")
        .SetTags("debug")
        .SetDescription("Verbose network events"),
    perfetto::Category("audio.latency")
        .SetTags("verbose")
        .SetDescription("Detailed audio latency metrics"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#include <chrono>
#include <fstream>
#include <thread>

namespace 
{
void InitializePerfetto() 
{
    perfetto::TracingInitArgs args;
    // The backends determine where trace events are recorded. For this example we
    // are going to use the in-process tracing service, which only includes in-app
    // events.
    args.backends = perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
}

std::unique_ptr<perfetto::TracingSession> StartTracing() 
{
    // The trace config defines which types of data sources are enabled for
    // recording. In this example we just need the "track_event" data source,
    // which corresponds to the TRACE_EVENT trace points.
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(1024);
    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    auto tracing_session = perfetto::Tracing::NewTrace();
    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();
    return tracing_session;
}

void StopTracing(std::unique_ptr<perfetto::TracingSession> tracing_session) 
{
    // Make sure the last event is closed for this example.
    perfetto::TrackEvent::Flush();
    // Stop tracing and read the trace data.
    tracing_session->StopBlocking();
    std::vector<char> trace_data(tracing_session->ReadTraceBlocking());
    // Write the result into a file.
    // Note: To save memory with longer traces, you can tell Perfetto to write
    // directly into a file by passing a file descriptor into Setup() above.
    std::ofstream output;
    output.open("example.pftrace", std::ios::out | std::ios::binary);
    output.write(&trace_data[0], std::streamsize(trace_data.size()));
    output.close();
    PERFETTO_LOG(
        "Trace written in example.pftrace file. To read this trace in "
        "text form, run `./tools/traceconv text example.pftrace`");
}

void DrawPlayer(int player_number) 
{
    TRACE_EVENT("rendering", "DrawPlayer", "player_number", player_number);
    {
        TRACE_EVENT("rendering", "DrawPlayer", "player_number", player_number);
    }
    // Sleep to simulate a long computation.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void DrawGame() 
{
    // This is an example of an unscoped slice, which begins and ends at specific
    // points (instead of at the end of the current block scope).
    TRACE_EVENT_BEGIN("rendering", "DrawGame");
    DrawPlayer(1);
    DrawPlayer(2);
    TRACE_EVENT_END("rendering");
    // Record the rendering framerate as a counter sample.
    TRACE_COUNTER("rendering", "Framerate", 120);
}
}  // namespace

int main(int, const char**) 
{
    InitializePerfetto();
    auto tracing_session = StartTracing();
    // Give a custom name for the traced process.
    perfetto::ProcessTrack process_track = perfetto::ProcessTrack::Current();
    perfetto::protos::gen::TrackDescriptor desc = process_track.Serialize();
    desc.mutable_process()->set_process_name("Example");
    perfetto::TrackEvent::SetTrackDescriptor(process_track, desc);

    // Simulate some work that emits trace events.
    DrawGame();
    
    StopTracing(std::move(tracing_session));
    return 0;
}

