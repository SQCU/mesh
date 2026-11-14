/* design/compiler-native-assignment-2026-11-13.md; output_data/compiler-workgroup-20261113/executor/aoti-binding-contract.md */
/* torch 2.14 torch/csrc/inductor/aoti_package/model_package_loader.h AOTIModelPackageLoader, boxed_run */
/* torch 2.14 torch/csrc/inductor/aoti_torch/tensor_converter.h */
#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

using torch::inductor::AOTIModelPackageLoader;

static void report(const char *text, char *error, size_t room) {
    if (room) strlcpy(error, text, room);
}

extern "C" void *mesh_aoti_load(const char *package_path, const char *model_name, char *error, size_t room) {
    try {
        return new AOTIModelPackageLoader(package_path, model_name);
    } catch (const std::exception &failure) {
        report(failure.what(), error, room);
    } catch (...) {
        report("AOTIModelPackageLoader: a non-standard exception", error, room);
    }
    return nullptr;
}

/* aoti_runtime/interface.h AOTInductorModelContainerRun: inputs stolen, outputs owned by the caller */
extern "C" int mesh_aoti_run(void *model, void **inputs, size_t input_count, void **outputs, size_t output_count,
                             char *error, size_t room) {
    auto *given = reinterpret_cast<AtenTensorHandle *>(inputs);
    try {
        std::vector<at::Tensor> results = static_cast<AOTIModelPackageLoader *>(model)->boxed_run(
            torch::aot_inductor::alloc_tensors_by_stealing_from_handles(given, input_count));
        if (results.size() == output_count) {
            std::vector<AtenTensorHandle> made = torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(results);
            std::copy(made.begin(), made.end(), reinterpret_cast<AtenTensorHandle *>(outputs));
            return 0;
        }
        report(("boxed_run returned " + std::to_string(results.size()) + " outputs for " +
                std::to_string(output_count) + " output slots").c_str(), error, room);
    } catch (const std::exception &failure) {
        report(failure.what(), error, room);
    } catch (...) {
        report("boxed_run: a non-standard exception", error, room);
    }
    std::vector<AtenTensorHandle> unstolen(given, given + input_count);
    torch::aot_inductor::free_unstolen_handles(unstolen);
    std::fill(given, given + input_count, nullptr);
    return 1;
}

extern "C" void mesh_aoti_free(void *model) {
    delete static_cast<AOTIModelPackageLoader *>(model);
}
