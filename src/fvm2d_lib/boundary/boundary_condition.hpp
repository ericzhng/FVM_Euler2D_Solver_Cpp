#pragma once

#include "common/fvm_export.hpp"
#include "core/types.hpp"
#include "physics/physics_model.hpp"
#include <vector>
#include <map>
#include <string>

namespace fvm2d {

/**
 * @brief Boundary condition specification for a patch
 */
struct FVM_API BoundarySpec {
    BCType type = BCType::Transmissive;
    VectorXd values;  // Primitive values for the BC

    BoundarySpec() = default;
    BoundarySpec(BCType t, const VectorXd& v) : type(t), values(v) {}
};

/**
 * @brief Boundary condition lookup table
 *
 * Provides efficient lookup of boundary conditions by tag ID
 */
class FVM_API BoundaryConditionLookup {
public:
    BoundaryConditionLookup() = default;

    /**
     * @brief Add a boundary condition for a patch
     * @param patch_name Name of the boundary patch
     * @param spec Boundary specification
     */
    void add(const std::string& patch_name, const BoundarySpec& spec);

    /**
     * @brief Set the boundary patch map from mesh
     * @param patch_map Map from patch name to tag ID
     */
    void set_patch_map(const std::map<std::string, int>& patch_map);

    /**
     * @brief Build the lookup table (call after adding all BCs)
     * @param num_vars Number of solution variables
     */
    void build(int num_vars);

    /**
     * @brief Apply boundary condition to get ghost cell state
     * @param tag Boundary tag ID
     * @param U_inside Interior cell state
     * @param normal Face normal
     * @param physics Physics model for BC application
     * @return Ghost cell state
     */
    VectorXd apply(int tag, const VectorXd& U_inside, const Vector2d& normal,
                   const PhysicsModel& physics) const;

    /**
     * @brief Get BC type for a tag
     */
    BCType type(int tag) const;

    /**
     * @brief Get BC values for a tag
     */
    const VectorXd& values(int tag) const;

private:
    std::map<std::string, BoundarySpec> specs_;
    std::map<std::string, int> patch_map_;

    // Lookup table indexed by tag
    std::vector<BCType> types_;
    std::vector<VectorXd> values_lookup_;

    int max_tag_ = 0;
};

/**
 * @brief Apply a single boundary condition
 *
 * @param U_inside Interior cell state
 * @param normal Face normal
 * @param bc_type Boundary condition type
 * @param bc_value Boundary condition values
 * @param physics Physics model
 * @return Ghost cell state
 */
FVM_API VectorXd apply_boundary_condition(
    const VectorXd& U_inside,
    const Vector2d& normal,
    BCType bc_type,
    const VectorXd& bc_value,
    const PhysicsModel& physics
);

}  // namespace fvm2d
