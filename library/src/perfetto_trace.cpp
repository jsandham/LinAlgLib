//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//********************************************************************************

#include <perfetto.h>

#include <chrono>
#include <fstream>
#include <thread>
#include <iostream>

#include "trace.h"

// The set of track event categories that the example is using.
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("linalg")
        .SetDescription("Rendering and graphics events"));

class perfetto_trace
{
    private:
        std::unique_ptr<perfetto::TracingSession> m_tracing_session;
        std::string m_output_filename;
        
        perfetto_trace();
        ~perfetto_trace();

        std::unique_ptr<perfetto::TracingSession> start_tracing();
        void stop_tracing(std::unique_ptr<perfetto::TracingSession> tracing_session);

    public:
        perfetto_trace(perfetto_trace const&)          = delete;
        void operator=(perfetto_trace const&) = delete;

        static perfetto_trace& get_instance()
        {
            static perfetto_trace instance;
            return instance;
        }

        void push(const char* name)
        {
            TRACE_EVENT_BEGIN("linalg", perfetto::StaticString{name});
        }

        void pop()
        {
            TRACE_EVENT_END("linalg");
        }
};

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

static void initialize_perfetto() 
{
    perfetto::TracingInitArgs args;
    // The backends determine where trace events are recorded. For this example we
    // are going to use the in-process tracing service, which only includes in-app
    // events.
    args.backends = perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
}

std::unique_ptr<perfetto::TracingSession> perfetto_trace::start_tracing() 
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

void perfetto_trace::stop_tracing(std::unique_ptr<perfetto::TracingSession> tracing_session) 
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
    output.open(m_output_filename.c_str(), std::ios::out | std::ios::binary);
    output.write(&trace_data[0], std::streamsize(trace_data.size()));
    output.close();
    PERFETTO_LOG(
        "Trace written in example.pftrace file. To read this trace in "
        "text form, run `./tools/traceconv text example.pftrace`");
}

perfetto_trace::perfetto_trace()
{
    initialize_perfetto();
    m_tracing_session = start_tracing();
    m_output_filename = "example.pftrace";
};
perfetto_trace::~perfetto_trace()
{
    stop_tracing(std::move(m_tracing_session));
};

trace::trace(const char* name)
{
    perfetto_trace::get_instance().push(name);
}
trace::~trace()
{
    perfetto_trace::get_instance().pop();
}
