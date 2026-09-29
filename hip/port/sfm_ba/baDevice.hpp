// This file is part of the Cheshire patch set for AliceVision (https://github.com/dspl1236/Cheshire).
// Copied by scripts/apply_hip_patch.py (step 7f) to src/aliceVision/sfm/bundle/gpu/.
//
// 0.3.7, step 4c of Cheshire's own bundle-adjustment solver (docs/notes/ba-own-solver.md): the solver's work
// per row and per landmark on the device. The host solver (ownSolver.hpp) keeps the trust-region loop, the
// vectors and the factorisation of the reduced camera system; the device holds the rows, the Jacobian and
// the residuals, and runs
//   - the evaluation (residuals, Jacobians, the loss, the manifolds' products, the cost and the gradient),
//   - the column norms, the Jacobi scaling and the model cost change,
//   - the landmark elimination into the reduced camera system, and the back-substitution.
// Every value is computed by the host's arithmetic (baArith.hpp, no contraction) and every sum in the host's
// order, so the result is the host solver's, byte for byte. No CUDA type appears here; baDevice.cu is the
// CUDA dialect (HIP through Cheshire's compat header).
#pragma once

#include <aliceVision/sfm/bundle/costfunctions/baArith.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aliceVision {
namespace sfm {
namespace cheshire {
namespace device {

// A device the solver can use (and CHESHIRE_BA_DEVICE is not 0); its name when there is one
bool available(std::string* name = nullptr);

// One row of the direct build (intrinsics, distortion, pose, landmark; two residuals), in the solver's order
struct Row
{
    std::int64_t j = 0;       // the row's first Jacobian value
    std::int32_t pj[4] = {-1, -1, -1, -1};      // per slot, its tangent block's offset from j, or -1 (constant)
    std::int32_t blk[4] = {-1, -1, -1, -1};     // slots 0-2: the F block; slot 3: the E block (the landmark); -1 constant
    std::int32_t manif[4] = {-1, -1, -1, -1};   // per slot, its manifold in the manifold table, or -1
    std::int32_t pose = 0;    // into the pose table
    std::int32_t camera = 0;  // into the camera table
    std::int32_t lm = 0;      // a free landmark: its E index; a constant one: -1 - its index in Structure::constLandmarks
    std::int32_t pad = 0;
    double ox = 0, oy = 0, os = 0;   // the observation: x, y, scale
    double weight = 1.0;             // ScaledLoss's factor (1: none)
};

struct Camera
{
    std::int32_t hasDistortion = 0;
    std::int32_t nd = 1;          // the distortion block's size
    std::int32_t isize = 4;       // the intrinsics block's size
    std::int32_t pad = 0;
};

// Per pose and evaluation: the rotation's part per pose (host, libm) and the centre
struct PoseEntry
{
    arith::PoseRotation rot;
    double center[3] = {0, 0, 0};
};

struct Structure
{
    std::vector<Row> rows;
    std::vector<std::int32_t> chunkStart;   // nE + 2: per E block its first row; [nE] the first row without one; [nE + 1] the end
    std::vector<std::int32_t> parts;        // the evaluation partitions' row boundaries
    std::vector<std::int32_t> groups;       // the assembly groups' chunk boundaries
    int nE = 0;
    int ne = 0, nf = 0, n = 0;              // tangent columns: E, F, both
    // F blocks
    std::vector<std::int32_t> fTsize, fCol;  // tangent size, column offset from ne
    // per chunk its F blocks (sorted) and their E^T F blocks' offsets
    std::vector<std::int64_t> cfStart, cfM;
    std::vector<std::int32_t> cf;
    std::int64_t mValues = 0;
    // per F block its chunks in order (c >= 0), then the rows without an E block (-1 - row); fcChunks[f] the
    // number of chunk entries; fcPos the block's position in the chunk's list
    std::vector<std::int64_t> fcStart;
    std::vector<std::int32_t> fc, fcPos, fcChunks;
    // the reduced camera system, upper block triangle: row f1 holds the blocks (f1, f2 >= f1), each
    // t1 x t2 row-major at sOff
    std::vector<std::int64_t> sStart, sOff;
    std::vector<std::int32_t> sCol;
    std::int64_t sValues = 0, jValues = 0;
    std::vector<double> constLandmarks;     // 3 per constant landmark
    std::vector<Camera> cameras;
    int nPoses = 0;
    std::vector<std::int32_t> manifSize, manifTsize;   // per manifold table entry
    bool loss = true;                        // Huber; false: no loss
    double hubA = 0.0, hubB = 0.0;
};

enum class Mode
{
    Full = 0,      // residuals and Jacobian kept on the device, the cost, the gradient
    Cost = 1,      // the cost alone
    Gradient = 2   // the cost and the gradient; the kept residuals and Jacobian untouched
};

class Problem
{
  public:
    // Uploads the structure; null (and why) when the device cannot take it
    static std::unique_ptr<Problem> create(const Structure& s, std::string* why);
    ~Problem();

    // One evaluation at the landmarks xE (3 nE values, E order); the poses', cameras' and manifolds' values of
    // this evaluation. Per partition its cost and whether every row evaluated; the gradient (n) when asked.
    bool evaluate(Mode mode, const double* xE, const PoseEntry* poses, const arith::CameraValues* cameras, const double* plusJacobians,
                  double* partCost, char* partOk, double* gradient);
    bool squaredColumnNorms(double* out);                      // n
    bool scaleColumns(const double* scale);                    // n; the kept Jacobian's columns
    bool modelCostChange(const double* step, double* partDot); // step n; per partition
    // The reduced camera system at the LM diagonal D (n): S's values and rhs (nf); false in *finite when an
    // inverse is not finite
    bool eliminate(const double* D, double* sValues, double* rhs, bool* finite);
    // y's E part (ne) from the camera part yf (nf); false in *finite when a value is not finite
    bool backSubstitute(const double* yf, double* yE, bool* finite);

    std::string error() const;
    double seconds(int phase) const;   // device time per phase (see baDevice.cu)

    struct Impl;

  private:
    explicit Problem(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> _impl;
};

}  // namespace device
}  // namespace cheshire
}  // namespace sfm
}  // namespace aliceVision
