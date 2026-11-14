# Non-CPU complex eig: connected source and actual output

Source checkpoints983afe62/646dda95/cba469af/41e1b2fd implement balancing, active Hessenberg reduction/Q, complex QR and scaled right eigenvector recovery. bc2cb879 connects GEEV scaling/backtransform/normalization through the placed nonCPU complex eig/eigvals source stage and fake-only target. 7f4bc3aa fixes the actual complex scatter-only record: one full real-pair destination, no duplicate ordinary output store.

The first public dense3×3 complex64 MPS call compiled, archived, loaded and invoked, but failed numerically (residual0.4599). The source discrepancy explained an extra ordinary store to component[0,0]; no native workaround or input change was used.

The identical public call after correction ended0/stderr0. Eigenvalue error1.7858e-7, relative eigenpair residual1.8001e-7, maximum column norm error2.1325e-7, largest-component imaginary part0, all results finite. The actual native artifact restored3rows with no relation solve. Exact arrays, hashes and logs are in nonsymmetric-native-result.json and nonsymmetric-native-2.stdout.

This establishes one dense complex64 MPS eig case. Values-only, batches, empty/order1, complex128, nonfinite/error exits and ANE behavior have source paths but no runtime evidence here. Real nonsymmetric input retains its prior CPU LAPACK path; there is no nonCPU real GEEV implementation. ZGEHD2/ZTREVC are unblocked and ZLAHQR is used without large-matrix multishift/AED or rescue; no large-matrix efficiency claim. Reduction order, Smith complex division and max-scaled norms retain the documented finite-precision distinctions.
