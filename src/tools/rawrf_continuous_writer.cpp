#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace bip = boost::interprocess;
namespace pt = boost::property_tree;

struct RingbufferStatus {
  uint64_t version;
  uint64_t write_sample;
  double rx_rate;
  double stream_start_time;
};

struct Args {
  std::string config_path;
  std::string ringbuffer_name;
  std::string output_dir = "/data/drf_data/rawrf_c";
  double chunk_seconds = 0.1;
  double safety_seconds = 0.05;
  double poll_seconds = 0.02;
  double duration_seconds = 0.0;
  double log_interval_seconds = 5.0;
  uint64_t decimate = 1;
  bool dry_run = false;
};

static std::atomic<bool> g_running(true);

void handle_sigint(int) { g_running.store(false); }

static std::string get_env_or_empty(const char *name) {
  const char *val = std::getenv(name);
  if (!val) {
    return std::string();
  }
  return std::string(val);
}

static bool starts_with(const std::string &s, const std::string &prefix) {
  return s.rfind(prefix, 0) == 0;
}

static void usage(const char *argv0) {
  std::cerr << "usage: " << argv0
            << " [--config PATH] [--ringbuffer-name NAME]"
            << " [--output-dir DIR] [--chunk-seconds S]"
            << " [--safety-seconds S] [--poll-seconds S]"
            << " [--duration-seconds S] [--log-interval S]"
            << " [--dry-run]" << std::endl;
}

static Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto take_value = [&](const std::string &key, std::string *out) {
      if (arg == key && i + 1 < argc) {
        *out = argv[++i];
        return true;
      }
      if (starts_with(arg, key + "=")) {
        *out = arg.substr(key.size() + 1);
        return true;
      }
      return false;
    };
    auto take_double = [&](const std::string &key, double *out) {
      std::string val;
      if (take_value(key, &val)) {
        *out = std::stod(val);
        return true;
      }
      return false;
    };
    auto take_uint64 = [&](const std::string &key, uint64_t *out) {
      std::string val;
      if (take_value(key, &val)) {
        *out = static_cast<uint64_t>(std::stoull(val));
        return true;
      }
      return false;
    };

    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (take_value("--config", &args.config_path)) {
      continue;
    } else if (take_value("--ringbuffer-name", &args.ringbuffer_name)) {
      continue;
    } else if (take_value("--output-dir", &args.output_dir)) {
      continue;
    } else if (take_double("--chunk-seconds", &args.chunk_seconds)) {
      continue;
    } else if (take_double("--safety-seconds", &args.safety_seconds)) {
      continue;
    } else if (take_double("--poll-seconds", &args.poll_seconds)) {
      continue;
    } else if (take_double("--duration-seconds", &args.duration_seconds)) {
      continue;
    } else if (take_double("--log-interval", &args.log_interval_seconds)) {
      continue;
    } else if (take_uint64("--decimate", &args.decimate)) {
      continue;
    } else if (arg == "--dry-run") {
      args.dry_run = true;
      continue;
    } else {
      std::cerr << "unknown argument: " << arg << std::endl;
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (args.decimate == 0) {
    std::cerr << "--decimate must be >= 1" << std::endl;
    std::exit(1);
  }
  return args;
}

static std::string default_config_path() {
  std::string radar_id = get_env_or_empty("RADAR_ID");
  std::string base = get_env_or_empty("BOREALISPATH");
  if (radar_id.empty() || base.empty()) {
    return std::string();
  }
  std::ostringstream oss;
  oss << base << "/config/" << radar_id << "/" << radar_id
      << "_config.ini";
  return oss.str();
}

static void ensure_dir(const std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    if (S_ISDIR(st.st_mode)) {
      return;
    }
    std::cerr << "path exists and is not a directory: " << path << std::endl;
    std::exit(1);
  }
  if (mkdir(path.c_str(), 0775) != 0) {
    std::perror("mkdir");
    std::cerr << "failed to create dir: " << path << std::endl;
    std::exit(1);
  }
}

static void parse_rx_channels(const pt::ptree &config,
                              std::vector<int> &rx_main,
                              std::vector<int> &rx_intf,
                              std::string &ringbuffer_name_out) {
  rx_main.clear();
  rx_intf.clear();
  ringbuffer_name_out = config.get<std::string>("ringbuffer_name");

  auto n200_list = config.get_child("n200s");
  for (const auto &n200 : n200_list) {
    std::string rx0 = n200.second.get<std::string>("rx_channel_0", "");
    std::string rx1 = n200.second.get<std::string>("rx_channel_1", "");

    auto handle_channel = [&](const std::string &ch) {
      if (ch.empty()) {
        return;
      }
      if (ch.size() < 2) {
        throw std::runtime_error("invalid channel string: " + ch);
      }
      char prefix = ch[0];
      int ant = std::stoi(ch.substr(1));
      if (prefix == 'm') {
        rx_main.push_back(ant);
      } else if (prefix == 'i') {
        rx_intf.push_back(ant);
      } else {
        throw std::runtime_error("invalid channel prefix: " + ch);
      }
    };

    handle_channel(rx0);
    handle_channel(rx1);
  }

  std::sort(rx_main.begin(), rx_main.end());
  rx_main.erase(std::unique(rx_main.begin(), rx_main.end()), rx_main.end());
  std::sort(rx_intf.begin(), rx_intf.end());
  rx_intf.erase(std::unique(rx_intf.begin(), rx_intf.end()), rx_intf.end());
}

static RingbufferStatus read_status(const volatile RingbufferStatus *status) {
  RingbufferStatus snap{};
  for (int attempt = 0; attempt < 8; ++attempt) {
    uint64_t v1 = status->version;
    std::atomic_thread_fence(std::memory_order_acquire);
    snap.write_sample = status->write_sample;
    snap.rx_rate = status->rx_rate;
    snap.stream_start_time = status->stream_start_time;
    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t v2 = status->version;
    if (v1 == v2 && (v1 % 2 == 0)) {
      snap.version = v2;
      return snap;
    }
    usleep(1000);
  }
  snap.version = status->version;
  snap.write_sample = status->write_sample;
  snap.rx_rate = status->rx_rate;
  snap.stream_start_time = status->stream_start_time;
  return snap;
}

int main(int argc, char **argv) {
  std::signal(SIGINT, handle_sigint);
  std::signal(SIGTERM, handle_sigint);

  Args args = parse_args(argc, argv);

  std::string config_path = args.config_path.empty() ? default_config_path()
                                                     : args.config_path;
  if (config_path.empty()) {
    std::cerr << "config path not provided and RADAR_ID/BOREALISPATH not set"
              << std::endl;
    return 1;
  }

  pt::ptree config;
  try {
    pt::read_json(config_path, config);
  } catch (const std::exception &ex) {
    std::cerr << "failed to read config: " << ex.what() << std::endl;
    return 1;
  }

  std::vector<int> rx_main;
  std::vector<int> rx_intf;
  std::string ringbuffer_name_cfg;
  try {
    parse_rx_channels(config, rx_main, rx_intf, ringbuffer_name_cfg);
  } catch (const std::exception &ex) {
    std::cerr << "failed to parse rx channels: " << ex.what() << std::endl;
    return 1;
  }

  std::string ringbuffer_name =
      args.ringbuffer_name.empty() ? ringbuffer_name_cfg : args.ringbuffer_name;
  if (ringbuffer_name.empty()) {
    std::cerr << "ringbuffer name not set" << std::endl;
    return 1;
  }

  std::vector<std::string> channel_names;
  for (int ant : rx_main) {
    channel_names.push_back("m" + std::to_string(ant));
  }
  for (int ant : rx_intf) {
    channel_names.push_back("i" + std::to_string(ant));
  }

  if (channel_names.empty()) {
    std::cerr << "no rx channels configured" << std::endl;
    return 1;
  }

  std::string status_name = ringbuffer_name + "_status";
  bip::shared_memory_object status_shm;
  while (true) {
    try {
      status_shm = bip::shared_memory_object(bip::open_only, status_name.c_str(),
                                             bip::read_only);
      break;
    } catch (...) {
      std::cerr << "waiting for shared memory: " << status_name << std::endl;
      usleep(static_cast<useconds_t>(args.poll_seconds * 1e6));
    }
  }

  bip::mapped_region status_region(status_shm, bip::read_only);
  auto *status =
      static_cast<volatile RingbufferStatus *>(status_region.get_address());

  bip::shared_memory_object ring_shm;
  while (true) {
    try {
      ring_shm = bip::shared_memory_object(bip::open_only,
                                           ringbuffer_name.c_str(),
                                           bip::read_only);
      break;
    } catch (...) {
      std::cerr << "waiting for shared memory: " << ringbuffer_name << std::endl;
      usleep(static_cast<useconds_t>(args.poll_seconds * 1e6));
    }
  }

  bip::mapped_region ring_region(ring_shm, bip::read_only);
  size_t region_size = ring_region.get_size();
  size_t channel_count = channel_names.size();
  size_t elem_size = sizeof(std::complex<float>);
  if (region_size % (channel_count * elem_size) != 0) {
    std::cerr << "ringbuffer size is not divisible by channel count" << std::endl;
    return 1;
  }
  size_t ring_size = region_size / (channel_count * elem_size);

  const std::complex<float> *ringbuffer =
      static_cast<const std::complex<float> *>(ring_region.get_address());

  ensure_dir(args.output_dir);
  std::vector<int> fds;
  if (!args.dry_run) {
    for (const auto &name : channel_names) {
      std::string ch_dir = args.output_dir + "/" + name;
      ensure_dir(ch_dir);
      std::string file_path = ch_dir + "/rawrf_continuous.cf32";
      int fd = ::open(file_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0664);
      if (fd < 0) {
        std::perror("open");
        std::cerr << "failed to open " << file_path << std::endl;
        return 1;
      }
      fds.push_back(fd);
    }
  }

  std::cout << "rawrf_continuous_writer started" << std::endl;
  std::cout << "channels=" << channel_names.size() << " ring_size=" << ring_size
            << " samples" << std::endl;
  std::cout << "decimate=" << args.decimate << std::endl;

  uint64_t last_sample = 0;
  bool have_last = false;
  uint64_t overrun_count = 0;
  uint64_t bytes_written = 0;
  uint64_t chunk_index = 0;

  auto start_time = std::chrono::steady_clock::now();
  auto last_log = start_time;

  while (g_running.load()) {
    RingbufferStatus snap = read_status(status);
    if (snap.rx_rate <= 0.0) {
      usleep(static_cast<useconds_t>(args.poll_seconds * 1e6));
      continue;
    }

    uint64_t write_sample = snap.write_sample;
    double rx_rate = snap.rx_rate;

    if (!have_last) {
      last_sample = write_sample;
      have_last = true;
      usleep(static_cast<useconds_t>(args.poll_seconds * 1e6));
      continue;
    }

    uint64_t chunk_samples =
        std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(args.chunk_seconds * rx_rate)));
    uint64_t safety_samples =
        std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(args.safety_seconds * rx_rate)));
    if (chunk_samples > ring_size / 2) {
      chunk_samples = std::max<uint64_t>(1, ring_size / 2);
    }

    if (write_sample < last_sample) {
      last_sample = write_sample;
    }

    if (write_sample - last_sample > ring_size) {
      overrun_count++;
      last_sample = write_sample - ring_size / 2;
    }

    uint64_t target_sample = write_sample - safety_samples;
    while (last_sample + chunk_samples <= target_sample) {
      size_t start_idx = static_cast<size_t>(last_sample % ring_size);
      size_t first_len = std::min<size_t>(chunk_samples, ring_size - start_idx);
      size_t second_len = chunk_samples - first_len;

      bool do_write = (args.decimate == 1) || (chunk_index % args.decimate == 0);
      if (do_write && !args.dry_run) {
        for (size_t ch = 0; ch < channel_count; ++ch) {
          const std::complex<float> *base = ringbuffer + ch * ring_size + start_idx;
          ssize_t bytes1 = ::write(fds[ch], base, first_len * elem_size);
          if (bytes1 < 0) {
            std::perror("write");
            g_running.store(false);
            break;
          }
          bytes_written += static_cast<uint64_t>(bytes1);
          if (second_len > 0) {
            const std::complex<float> *base2 = ringbuffer + ch * ring_size;
            ssize_t bytes2 = ::write(fds[ch], base2, second_len * elem_size);
            if (bytes2 < 0) {
              std::perror("write");
              g_running.store(false);
              break;
            }
            bytes_written += static_cast<uint64_t>(bytes2);
          }
        }
      }

      last_sample += chunk_samples;
      chunk_index++;
    }

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last_log).count() >=
        args.log_interval_seconds) {
      double elapsed = std::chrono::duration<double>(now - start_time).count();
      double mbps = elapsed > 0 ? (bytes_written / 1e6) / elapsed : 0.0;
      std::cout << "elapsed=" << elapsed << "s" << " bytes_written="
                << bytes_written << " MB/s=" << mbps
                << " overruns=" << overrun_count << std::endl;
      last_log = now;
    }

    if (args.duration_seconds > 0.0) {
      double elapsed = std::chrono::duration<double>(now - start_time).count();
      if (elapsed >= args.duration_seconds) {
        break;
      }
    }

    usleep(static_cast<useconds_t>(args.poll_seconds * 1e6));
  }

  for (int fd : fds) {
    ::close(fd);
  }

  std::cout << "rawrf_continuous_writer exiting" << std::endl;
  return 0;
}
