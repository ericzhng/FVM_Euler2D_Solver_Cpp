#include "boundary/boundary_condition.hpp"
#include <stdexcept>
#include <algorithm>

namespace fvm2d {

void BoundaryConditionLookup::add(const std::string& patch_name, const BoundarySpec& spec) {
    specs_[patch_name] = spec;
}

void BoundaryConditionLookup::set_patch_map(const std::map<std::string, int>& patch_map) {
    patch_map_ = patch_map;

    // Find maximum tag
    max_tag_ = 0;
    for (const auto& [name, tag] : patch_map) {
        max_tag_ = std::max(max_tag_, tag);
    }
}

void BoundaryConditionLookup::build(int num_vars) {
    // Resize lookup tables
    types_.resize(max_tag_ + 1, BCType::Transmissive);
    values_lookup_.resize(max_tag_ + 1, VectorXd::Zero(num_vars));

    // Fill lookup tables
    for (const auto& [name, spec] : specs_) {
        auto it = patch_map_.find(name);
        if (it != patch_map_.end()) {
            int tag = it->second;
            types_[tag] = spec.type;
            if (spec.values.size() > 0) {
                values_lookup_[tag] = spec.values;
            }
        }
    }
}

VectorXd BoundaryConditionLookup::apply(int tag, const VectorXd& U_inside,
                                         const Vector2d& normal,
                                         const PhysicsModel& physics) const {
    if (tag < 0 || tag > max_tag_) {
        return physics.apply_transmissive_bc(U_inside);
    }

    return apply_boundary_condition(U_inside, normal, types_[tag],
                                     values_lookup_[tag], physics);
}

BCType BoundaryConditionLookup::type(int tag) const {
    if (tag < 0 || tag > max_tag_) {
        return BCType::Transmissive;
    }
    return types_[tag];
}

const VectorXd& BoundaryConditionLookup::values(int tag) const {
    static VectorXd empty;
    if (tag < 0 || tag > max_tag_) {
        return empty;
    }
    return values_lookup_[tag];
}

VectorXd apply_boundary_condition(
    const VectorXd& U_inside,
    const Vector2d& normal,
    BCType bc_type,
    const VectorXd& bc_value,
    const PhysicsModel& physics
) {
    switch (bc_type) {
        case BCType::Transmissive:
            return physics.apply_transmissive_bc(U_inside);

        case BCType::Inlet:
            return physics.apply_inlet_bc(bc_value);

        case BCType::Outlet:
            return physics.apply_outlet_bc(U_inside, bc_value);

        case BCType::Wall:
            return physics.apply_wall_bc(U_inside, normal, bc_value);

        default:
            return physics.apply_transmissive_bc(U_inside);
    }
}

}  // namespace fvm2d
