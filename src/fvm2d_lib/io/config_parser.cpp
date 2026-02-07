#include "core/config.hpp"
#include "boundary/boundary_condition.hpp"
#include <yaml-cpp/yaml.h>
#include <mpi.h>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace fvm2d {

SolverConfig parse_config(const std::string& filepath) {
    SolverConfig config;

    try {
        YAML::Node root = YAML::LoadFile(filepath);

        // Input section
        if (root["input"]) {
            if (root["input"]["mesh_dir"]) {
                config.mesh_dir = root["input"]["mesh_dir"].as<std::string>();
            } else if (root["input"]["mesh_file"]) {
                // Legacy: extract directory from mesh file path
                std::string mesh_file = root["input"]["mesh_file"].as<std::string>();
                size_t pos = mesh_file.find_last_of("/\\");
                config.mesh_dir = (pos != std::string::npos) ? mesh_file.substr(0, pos) : ".";
            }
            if (root["input"]["boundary_config"]) {
                config.boundary_config_file = root["input"]["boundary_config"].as<std::string>();
            }
        }

        // Simulation section
        if (root["simulation"]) {
            auto sim = root["simulation"];
            if (sim["t_end"]) config.t_end = sim["t_end"].as<Scalar>();
            if (sim["equation"]) {
                config.equation = parse_equation_type(sim["equation"].as<std::string>());
            }
            if (sim["case"]) config.case_name = sim["case"].as<std::string>();
        }

        // Physics section
        if (root["physics"]) {
            auto phys = root["physics"];
            if (phys["euler"] && phys["euler"]["gamma"]) {
                config.physics.gamma = phys["euler"]["gamma"].as<Scalar>();
            }
            if (phys["shallow_water"] && phys["shallow_water"]["g"]) {
                config.physics.g = phys["shallow_water"]["g"].as<Scalar>();
            }
        }

        // Solver section
        if (root["solver"]) {
            auto solver = root["solver"];

            // Time integration
            if (solver["time_integration"]) {
                auto time = solver["time_integration"];
                if (time["method"]) {
                    config.time.method = parse_time_integration(time["method"].as<std::string>());
                }
                if (time["cfl"]) config.time.cfl = time["cfl"].as<Scalar>();
                if (time["use_adaptive_dt"]) config.time.use_adaptive_dt = time["use_adaptive_dt"].as<bool>();
                if (time["dt_initial"]) config.time.dt_initial = time["dt_initial"].as<Scalar>();
            }

            // Spatial discretization
            if (solver["spatial"]) {
                auto spatial = solver["spatial"];
                if (spatial["flux_type"]) {
                    config.spatial.flux_type = parse_flux_type(spatial["flux_type"].as<std::string>());
                }
                if (spatial["limiter_type"]) {
                    config.spatial.limiter_type = parse_limiter_type(spatial["limiter_type"].as<std::string>());
                }
                if (spatial["gradient_over_relaxation"]) {
                    config.spatial.gradient_over_relaxation = spatial["gradient_over_relaxation"].as<Scalar>();
                }
            }
        }

        // Output section
        if (root["output"]) {
            auto output = root["output"];
            if (output["format"]) config.output.format = output["format"].as<std::string>();
            if (output["interval"]) config.output.interval = output["interval"].as<int>();
            if (output["filename_prefix"]) config.output.filename_prefix = output["filename_prefix"].as<std::string>();
            if (output["output_dir"]) config.output.output_dir = output["output_dir"].as<std::string>();
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing config file '" + filepath + "': " + e.what());
    }

    return config;
}

void broadcast_config(SolverConfig& config, void* comm_ptr) {
    MPI_Comm comm = *static_cast<MPI_Comm*>(comm_ptr);
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Serialize configuration to a string buffer for broadcast
    std::ostringstream oss;
    if (rank == 0) {
        // Write all config fields to stream
        auto write_string = [&oss](const std::string& s) {
            int len = static_cast<int>(s.size());
            oss.write(reinterpret_cast<const char*>(&len), sizeof(int));
            oss.write(s.data(), len);
        };

        write_string(config.mesh_dir);
        write_string(config.boundary_config_file);
        oss.write(reinterpret_cast<const char*>(&config.t_end), sizeof(Scalar));
        int eq = static_cast<int>(config.equation);
        oss.write(reinterpret_cast<const char*>(&eq), sizeof(int));
        write_string(config.case_name);

        // Time config
        int tm = static_cast<int>(config.time.method);
        oss.write(reinterpret_cast<const char*>(&tm), sizeof(int));
        oss.write(reinterpret_cast<const char*>(&config.time.cfl), sizeof(Scalar));
        int adaptive = config.time.use_adaptive_dt ? 1 : 0;
        oss.write(reinterpret_cast<const char*>(&adaptive), sizeof(int));
        oss.write(reinterpret_cast<const char*>(&config.time.dt_initial), sizeof(Scalar));

        // Spatial config
        int ft = static_cast<int>(config.spatial.flux_type);
        oss.write(reinterpret_cast<const char*>(&ft), sizeof(int));
        int lt = static_cast<int>(config.spatial.limiter_type);
        oss.write(reinterpret_cast<const char*>(&lt), sizeof(int));
        oss.write(reinterpret_cast<const char*>(&config.spatial.gradient_over_relaxation), sizeof(Scalar));

        // Output config
        write_string(config.output.format);
        oss.write(reinterpret_cast<const char*>(&config.output.interval), sizeof(int));
        write_string(config.output.filename_prefix);
        write_string(config.output.output_dir);

        // Physics config
        oss.write(reinterpret_cast<const char*>(&config.physics.gamma), sizeof(Scalar));
        oss.write(reinterpret_cast<const char*>(&config.physics.g), sizeof(Scalar));
    }

    // Broadcast buffer size and data
    std::string buffer = oss.str();
    int buffer_size = static_cast<int>(buffer.size());
    MPI_Bcast(&buffer_size, 1, MPI_INT, 0, comm);

    if (rank != 0) {
        buffer.resize(buffer_size);
    }
    MPI_Bcast(buffer.data(), buffer_size, MPI_CHAR, 0, comm);

    // Deserialize on non-root ranks
    if (rank != 0) {
        std::istringstream iss(buffer);

        auto read_string = [&iss]() -> std::string {
            int len;
            iss.read(reinterpret_cast<char*>(&len), sizeof(int));
            std::string s(len, '\0');
            iss.read(s.data(), len);
            return s;
        };

        config.mesh_dir = read_string();
        config.boundary_config_file = read_string();
        iss.read(reinterpret_cast<char*>(&config.t_end), sizeof(Scalar));
        int eq;
        iss.read(reinterpret_cast<char*>(&eq), sizeof(int));
        config.equation = static_cast<EquationType>(eq);
        config.case_name = read_string();

        // Time config
        int tm;
        iss.read(reinterpret_cast<char*>(&tm), sizeof(int));
        config.time.method = static_cast<TimeIntegrationType>(tm);
        iss.read(reinterpret_cast<char*>(&config.time.cfl), sizeof(Scalar));
        int adaptive;
        iss.read(reinterpret_cast<char*>(&adaptive), sizeof(int));
        config.time.use_adaptive_dt = (adaptive != 0);
        iss.read(reinterpret_cast<char*>(&config.time.dt_initial), sizeof(Scalar));

        // Spatial config
        int ft;
        iss.read(reinterpret_cast<char*>(&ft), sizeof(int));
        config.spatial.flux_type = static_cast<FluxType>(ft);
        int lt;
        iss.read(reinterpret_cast<char*>(&lt), sizeof(int));
        config.spatial.limiter_type = static_cast<LimiterType>(lt);
        iss.read(reinterpret_cast<char*>(&config.spatial.gradient_over_relaxation), sizeof(Scalar));

        // Output config
        config.output.format = read_string();
        iss.read(reinterpret_cast<char*>(&config.output.interval), sizeof(int));
        config.output.filename_prefix = read_string();
        config.output.output_dir = read_string();

        // Physics config
        iss.read(reinterpret_cast<char*>(&config.physics.gamma), sizeof(Scalar));
        iss.read(reinterpret_cast<char*>(&config.physics.g), sizeof(Scalar));
    }
}

std::map<std::string, BoundarySpec> parse_boundary_config(const std::string& filepath) {
    std::map<std::string, BoundarySpec> specs;

    try {
        YAML::Node root = YAML::LoadFile(filepath);

        if (!root["boundaries"]) {
            std::cerr << "Warning: No 'boundaries' section in " << filepath << std::endl;
            return specs;
        }

        auto boundaries = root["boundaries"];
        for (auto it = boundaries.begin(); it != boundaries.end(); ++it) {
            std::string name = it->first.as<std::string>();
            auto bc_node = it->second;

            BoundarySpec spec;

            // Parse type
            if (bc_node["type"]) {
                spec.type = parse_bc_type(bc_node["type"].as<std::string>());
            }

            // Parse values (optional)
            if (bc_node["values"]) {
                auto values_node = bc_node["values"];
                spec.values.resize(values_node.size());
                for (std::size_t i = 0; i < values_node.size(); ++i) {
                    spec.values[i] = values_node[i].as<Scalar>();
                }
            }

            specs[name] = spec;
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing boundary config '" + filepath + "': " + e.what());
    }

    return specs;
}

}  // namespace fvm2d
