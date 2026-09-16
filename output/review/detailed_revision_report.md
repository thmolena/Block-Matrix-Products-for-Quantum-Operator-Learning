Detailed manuscript review and submission strategy — September 15, 2026

Reviewed: all 903 lines of `main.tex`, all 15 rendered pages of `main.pdf`, generated numerical text, the principal certificate/Krylov/selection/learning implementations, tests, result validation, and repository documentation. Source locations below refer to this reviewed snapshot. SHA-256 hashes are in `reviewed_files_sha256.json`. The manuscript and implementation have not been edited. This report and its reproducer are review artifacts, not part of a proposed public release.

The paper has a coherent specialist contribution: a loss-specific compression bound combined with separate polynomial approximation bounds for responses and derivatives, used to select a finite-grid rank/depth plan. It is not ready to submit because a reproducible deflation defect breaks the implementation's connection to its mathematical assumptions. After repair and stronger validation, SIAM Journal on Scientific Computing is the best fit for the work currently demonstrated. Nature Computational Science is an ambitious, scope-relevant target requiring substantial additional scientific evidence. Neither a LaTeX format nor any revision plan can guarantee acceptance at Nature or another journal.

Evidence status: I checked the mathematical arguments directly and independently constructed a failing implementation example. The existing 25 tests pass under Python 3.13.12, NumPy 2.4.3, SciPy 1.17.1 with warnings treated as errors; the release validator also passes. I did not replay the complete PARSEC and 288-run experiment suite or verify every bibliography record against a publisher. Stored experimental results are identified as such below. A preliminary run using the system's unsupported Python 3.9 produced numerical warnings; it is not evidence of failure within the package's declared Python support. The substantive counterexample was reproduced in the supported environment.

**1. Submission recommendation and verified impact factors**

The following Nature Portfolio numbers are the publisher's **2025 Journal Impact Factors**, checked on September 15, 2026. They are not CiteScores or five-year impact factors. Scope and readiness assessments are my judgment of this manuscript.

| Journal | 2025 JIF | Assessment for this paper |
|---|---:|---|
| Nature | 56.1 | Highest JIF among the scope-adjacent original-research journals considered here; current evidence does not establish a broadly consequential scientific advance. Do not submit this version. |
| Nature Computational Science | 20.3 | Best high-impact stretch target for the computational-method direction; requires stronger novelty, reliability, and compelling application-level utility. |
| Nature Communications | 18.1 | A possible stretch after a substantial advance of field-wide significance; broader scope does not make the present evidence sufficient. |
| npj Quantum Information | 9.0 | Consider only if the work develops a meaningful quantum-information/measurement contribution beyond classical matrix-function calculations. |
| Communications Physics | 5.5 | Consider if the revised paper leads with a substantive physics result or capability. |
| SIAM Journal on Scientific Computing | Not verified from a current primary metrics page | Recommended first target after repairs and strengthened experiments; directly aligned with numerical methods and computational evidence. |
| SIAM Journal on Matrix Analysis and Applications | Not verified from a current primary metrics page | Alternative if the theory becomes the main advance, especially sharp bounds, generalization, or finite-precision analysis. |

Nature figures: [official Nature Portfolio journal metrics](https://www.nature.com/nature-portfolio/about-journals/journal-metrics). SISC describes its emphasis on numerical methods and scientific computation in its [journal scope](https://www.siam.org/publications/siam-journals/siam-journal-on-scientific-computing/); SIMAX covers matrix theory, analysis, applications, and computation in its [scope](https://www.siam.org/publications/siam-journals/siam-journal-on-matrix-analysis-and-applications/). These specialist venues should not be ranked solely by raw JIF across different disciplines.

Nature requires outstanding scientific importance and interdisciplinary interest; its technical-paper criterion concerns significant impacts on research communities. A polished numerical-method manuscript does not automatically meet that threshold. [Nature editorial criteria](https://www.nature.com/nature/for-authors/editorial-criteria-and-processes).

For Nature Computational Science, the appropriate ambition is a reliable method that makes a consequential computational task possible or materially improves its cost/accuracy tradeoff. Its stated scope concerns computational techniques, mathematical models, and their scientific applications. A new physical experiment is not universally necessary, but this manuscript currently supplies neither an application breakthrough nor a consistent end-to-end performance benefit. [Nature Computational Science journal information](https://www.nature.com/natcomputsci/journal-information).

**2. The submission-blocking implementation finding**

**2.1 Unpivoted QR filtering can lose a nonzero independent direction.** In `code/src/bmqol/krylov.py:90–98`, the residual block is factored by unpivoted QR and columns of Q are retained according to the diagonal of R. When a dependent column occurs before a later independent column, an arbitrary QR completion vector can absorb the later column. Dropping that completion vector can then destroy the range of the residual block. This is a structural rank-detection problem, not merely a tiny floating-point perturbation.

Use standard coordinate vectors in R^6 and set

\[
B=[e_4,e_5,e_6],\quad X=[e_1,e_1,e_2],\quad H=XB^T+BX^T.
\]

Then H is symmetric, HB=X, and K_2(H,B) has dimension five. The current routine returns a four-dimensional basis. The candidate block has rank two but the new block has width one; the discarded span residual is 1. The returned basis is exactly orthogonal in this example, yet its degree-two response moment error is 1. Thus the reported orthogonality defect cannot establish that the required Krylov space has been represented.

**2.2 The public evaluation routine can return an infeasible actual computation.** Set A(theta)=(1+theta)H, theta=0, |theta|<=0.5, D=H, one time t=1, and Y=0. The valid row-sum enclosure is rho=3. With full rank, requested tolerance 1e-8, and candidate depths (20,24,28), the supported-environment run produced:

| Quantity | Observed value |
|---|---:|
| Preflight selection bound | 2.5899315865684225e-14 |
| Returned block depth | 2 |
| Returned basis dimension | 4 |
| Evaluated certificate field | 32.154513044762034 |
| Computed scalar gradient | approximately -0.2178396181 |
| Independent exponential-Fréchet gradient | approximately -0.6724883315 |
| Absolute gradient discrepancy | 0.45464871341284085 |

`preflight_gradient` calculates an evaluated certificate after early termination but returns without requiring it to satisfy the requested tolerance (`adaptive.py:268–291`). The small planned bound therefore does not certify this returned answer. The large evaluated bound visibly signals a problem but is not enforced by this API. The theorem assumes V spans the exact stated Krylov space; that assumption fails here. The counterexample does not refute that theorem or establish that the stored PARSEC results have the same error.

**Required repair.** Use a range-preserving rank-revealing factorization of each residual block, such as an SVD or an appropriately handled pivoted QR, and reorthogonalize the retained basis against the previous space. Check the residual block's representation error, not only V^T V-I. Distinguish exact invariant termination, numerically discarded directions, and a requested depth that was not reached. Reject, safely fall back, or use a proved defect correction when the required assumptions cannot be justified. Never present an unreached-depth preflight bound as a certificate for the returned computation. An evaluated tail bound alone is also insufficient if its polynomial-exactness assumptions have already failed.

**Required regression tests.** Add the supplied six-dimensional example; exact partial dependence in different column orders; nearly dependent residual blocks; tiny but nonzero starting singular values; invariant spaces smaller than n; wide B; and deliberately inadequate depth grids. Verify span preservation, polynomial moments, independent gradients, returned status, and failure behavior. Rerun the complete experiments after the repair and regenerate every dependent table, plot, digest, and packaged copy.

The reproducer is `deflation_counterexample.py` in this directory. Run with `PYTHONPATH=code/src python output/review/deflation_counterexample.py`. Passing the present suite and result validator is useful but demonstrably does not cover this failure mode.

**3. Title, abstract, and scientific positioning — main.tex:17–80**

**3.1 Narrow the title or broaden the results.** “Matrix-function gradients” sounds more general than the demonstrated cosine specialization. Suggested specialist title: **“A priori rank–depth selection for block cosine gradients.”** An alternative retaining the certification emphasis is **“Exact-arithmetic error bounds for compressed block cosine gradients.”** Keep “certified” only with the scope made prominent and the implementation failures repaired. Do not put “quantum advantage,” “fast,” or general “Hamiltonian learning” into the title without new supporting results.

**3.2 Make the contribution hierarchy explicit.** The strongest candidate novelty is the cancellation of retained–retained loss-gradient contributions, followed by its integration with separate response/derivative bounds. SVD truncation, block Krylov projection, cosine expansions, Fréchet derivatives, polarization, and exhaustive minimization over a finite grid are supporting ingredients. The grid minimum is immediate once feasibility is defined; it should not be presented as an independent major optimization theorem.

**3.3 State the practical question positively.** The introduction should explain who needs a pre-execution accuracy budget, why current error-controlled methods do not already solve that task, and what is gained by reducing probe width. The current opening cites general matrix-function literature but provides little concrete evidence about the motivating inverse problem. Add one worked application formulation and a precise computational bottleneck.

**3.4 Replace revision-history language.** “Revised default,” “original certificate,” and “original 96-site failures” sound like an internal revision record. Use scientific labels: linear-compression/Taylor baseline, loss-aware/Chebyshev method, difficult measurement design, and redesigned measurement study. Preserve all negative results but explain their scientific role rather than their chronology.

**3.5 Fix the abstract's attribution of success.** The 288 runs include all four methods. Only 72 are joint-method runs. The current phrase connecting all 288 recoveries to active compression can be misread as a result specific to the proposed method. Say that the joint method recovered in 72/72 tested cases and that all three controls also recovered. These are selected simulated problems, not 72 statistically independent draws from a population of inverse problems.

**3.6 Qualify “reduced learning computations.”** It is true only in the looser-stationarity groups for the reported Krylov work. At the strict target the joint method uses more work, and it is slower in every learning group. Use a conditional statement and report the reversal. The abstract should not end on an unqualified work-reduction claim.

**3.7 Clarify the 60% comparison.** 120 to 48 is a comparison with the manuscript's own linear/Taylor joint baseline. The refined full-rank certified comparison is 128 to 48, or 62.5%. Neither number is a speedup over a best existing algorithm. “Per gradient evaluation on each benchmark” would avoid ambiguity about aggregation over four matrices.

**3.8 Shorten and focus the abstract.** Remove the complete factorial-design list and the historical recovery narrative. Retain the problem, new bound, rank/depth selection, one matched-certified result, one practical limitation, and the simulation scope. The draft below is a possible approximately 150-word direction, conditional on repair and successful revalidation:

> Correlated preparations create redundant work in inverse problems involving block cosine responses. We derive an exact-arithmetic gradient-error bound that combines loss-aware probe compression with separate Chebyshev remainders for responses and operator derivatives. For consistent targets, cancellation of the retained-block contribution makes the compression bound quadratic in the discarded probe norm. A finite search selects the least nominal operator work among feasible rank–depth pairs before constructing a Krylov basis. On four electronic-structure matrices with constructed probes, the refined rule uses 48 operator-vector applications, compared with 128 for full-rank certification. Its runtime nevertheless exceeds an executable incremental stopping heuristic. In simulated lattice inverse problems, the joint method recovers parameters in all 72 tested cases, as do the controls; stringent stationarity targets can eliminate work savings. These results characterize the benefits and costs of a priori error control, without establishing floating-point certification, global identifiability, or a general runtime advantage.

**3.9 Expand the nearest-work comparison.** Singh, Barros, and Li already study forward-only projected sensitivities and quantify omitted basis-variation terms. Explain precisely how a preflight bound on the original block-loss gradient differs from their treatment of scalar forms and the projected approximation. A comparison table should distinguish scalar/block output, compression, a priori/a posteriori information, original-gradient versus surrogate-derivative error, reorthogonalization, and implementation provenance. Do not imply that your polarization implementation reproduces their algorithm. [Their current preprint](https://arxiv.org/html/2605.12801v1).

**4. Objective and access assumptions — main.tex:82–115**

**4.1 Define all spaces and norms once.** State H0,Dj in R^(n×n) symmetric, theta in the parameter box, B in R^(n×b), Y_l in R^(b×b), real times, s>=1, and the Euclidean/Frobenius/spectral/maximum-row-sum norm conventions. Explicitly distinguish the number of observations s from any scalar differentiation variable in proofs.

**4.2 Define the physical observation early.** B^T cos(tA)B is the real part of a transition-amplitude block for real symmetric A and real preparations. It is not a probability block. If the quantum application remains prominent, describe the phase reference and interference/amplitude access needed. The current disclaimer is good but arrives late in the experimental section.

**4.3 Separate matrix-free application access from norm information.** The theorem needs a certified bound rho and bounds on ||Dj||. A generic black-box matvec interface does not automatically provide row sums. State that these bounds are supplied analytically or computed from accessible entries; the benchmarks use stored sparse matrices. “Before an application of A” should mean before the gradient evaluation's Krylov work, after targets and norm bounds have been supplied. Simulated target generation itself uses the operator and is excluded from that cost claim.

**4.4 State norm-bound substitutions.** Introduce d_j>=||Dj||_2 as executable bound inputs and write the certificate using d_j. This avoids suggesting that general sparse spectral norms were computed exactly when row sums are actually used.

**4.5 Explain the loss normalization.** The objective averages over times, not over b^2 entries or parameter scales. As b changes, the absolute gradient tolerance need not represent the same relative task. Define absolute, relative, and direction errors with formulas and a stable convention when ||g|| is small. For box widths a_j that differ, discuss whether stationarity is measured in physical or normalized parameter coordinates.

**4.6 Give the derivative definition.** Add f(A+hD)=f(A)+h L_f(A,D)+O(h^2), and state why differentiation through the finite sum and Frobenius loss gives Eq. (2.2). This makes the paper accessible without a long matrix-function tutorial.

**4.7 Preserve the uniform-enclosure lemma.** The argument is correct for symmetric H0 and Dj. Retain its distinction between exact arithmetic and floating-point row sums. Record the benchmark scaling factor for every input, not just the resulting row-sum norm. Use the dimensionless product |t|rho to compare approximation difficulty.

**5. Compression theory — main.tex:118–185**

**5.1 Define the compressed objective explicitly.** Introduce F_l^(r)=B_r^T cos(t_l A)B_r, Phi_r=(1/(2s))sum ||F_l^(r)-Y_l||_F^2, and g^(r)=grad Phi_r. Theorem 2 currently uses g^(r) before formally defining it. Make clear that the original b×b targets are retained.

**5.2 State the admissible SVD ranks carefully.** For nonzero B, use 1<=r<=rank(B) for executable exact-arithmetic factorizations and distinguish a numerical-rank threshold. The current condition “nonzero retained singular values” is not sufficient for the implementation's relative QR threshold. Specify B=0 separately: the exact gradient is zero and no Krylov calculation is needed. Define full-rank behavior for rank-deficient and wide blocks.

**5.3 Keep the elementary compression bound but reduce its prominence.** The proof is sound under its stated assumptions. It is useful as a comparator and building block, but the loss-aware theorem deserves the main attention. The equality ||B_r||_2=||B||_2 assumes a nonempty retained SVD including the leading singular value; write that explicitly or keep an inequality that covers edge cases.

**5.4 Show the cancellation identity before bounding it.** In right-singular coordinates, write the exact difference as

\[
g_j-g_j^{(r)}=\frac1s\sum_\ell\left[
2\langle F_{\ell,rd}-Y_{\ell,rd},J_{\ell j,rd}\rangle_F+
\langle F_{\ell,dd}-Y_{\ell,dd},J_{\ell j,dd}\rangle_F\right].
\]

This makes the factor two and the missing retained–retained term transparent. The bound then follows directly. State symmetry or replace Y with sym(Y).

**5.5 State the precise asymptotic dependence.** Since u_r=||B||_2 delta_r and v_r=||C_d||_2 delta_r, consistent-target bounds have an O(delta_r^2) leading term for bounded ||B||. The discarded–discarded contribution is even smaller when ||C_d||_2<=delta_r. Do not imply quadratic dependence under arbitrary fixed additive noise.

**5.6 Add a noisy-target corollary.** If Y_l=F_l(theta_star)+N_l, the bracket is bounded by

\[
4u_r^2+2v_r^2+2u_r\|N_{\ell,rd}\|_F+v_r\|N_{\ell,dd}\|_F.
\]

This follows from the triangle inequality and exposes the potential O(delta_r ||N||) cross contribution. It connects the mathematical claim to the noise experiment and explains why aggressive compression may cease to help near stationarity. This is a proposed addition, not a result already demonstrated in the manuscript's experiments.

**5.7 Show projector equivalences.** Explain ||W_r^T Y W_d||_F=||W_r^T Y P_d||_F and ||W_d^T Y W_d||_F=||P_d Y P_d||_F. Include dimensions and the treatment of null singular directions for b>n.

**5.8 Add a sharpness experiment.** Vary a controlled discarded singular scale over several decades, plot actual compression-gradient error and both bounds, and report log-log slopes for clean and noisy targets. The existing quadratic-scaling unit test is useful but does not replace an explanatory scientific figure. Include at least one noncommuting A,D example and one poorly compressible block.

**6. Polynomial exactness and invariance — main.tex:187–233**

**6.1 Preserve the distinction between the two degrees.** The response proof through 2m-1 and derivative proof through m are correct for an orthonormal basis spanning the required Krylov space. Make the dependence on exact span explicit in Theorem 3 rather than relying entirely on surrounding text.

**6.2 Add an explicit counterexample beyond derivative exactness.** For m=2, take B=e1, A the adjacency matrix of a three-vertex path, and D=e1 e3^T+e3 e1^T. Then K2=span(e1,e2), projected D=0, but e1^T L_(x^3)(A,D)e1=2. This gives readers a concrete reason not to transfer response degree 2m-1 to derivative degree. For cosine, explain separately that only even polynomial degrees occur.

**6.3 Clarify that invariance suffices without D-invariance.** The sandwich identity follows because every A^k C_r remains in the subspace. Dj need not preserve that subspace. That useful fact deserves a sentence.

**6.4 Do not equate a square floating-point basis with exact certification.** In the exact theorem, square and orthogonal implies exact similarity. In implementation, orthogonality, projections, eigensolves, and summations still have errors. The current caveat is appropriate but should accompany all exported certificate/status terminology.

**6.5 Formalize deflation semantics.** Define m as Krylov polynomial depth, k as realized dimension, and block widths r_i separately. Exact dependence can be removed while preserving the space. Numerical truncation requires a separate perturbation analysis or an explicitly unverified status. Merely replacing m by the number of returned blocks is not enough if the wrong span was retained.

**7. Chebyshev tails and preflight theorem — main.tex:235–299 and Appendix A**

**7.1 The operator-derivative argument is a strength.** The recurrence identity in Eq. (3.9) gives the nu^2 bound in operator norm; it is not an unjustified application of a scalar derivative inequality. Add the short induction or recurrence derivation so readers can verify the noncommuting placement of D. The Bessel majorant is consistent with the real-argument specialization of [DLMF 10.14.4](https://dlmf.nist.gov/10.14.E4); the expansion is supported by [DLMF 10.12](https://dlmf.nist.gov/10.12).

**7.2 Make the refined theorem the principal statement.** Use generic response and derivative majorants e0,e1, or write the Chebyshev version first. Present Taylor as the baseline. At present the central theorem uses R and only afterward says E can be substituted, which obscures the actual method.

**7.3 Separate mathematical infinite tails from executable geometric bounds.** Give different symbols to the exact positive series and their evaluated upper majorants. State q=the first even integer exceeding the truncation degree, the denominator condition, zero-time/radius handling, and overflow behavior. A geometric denominator failure means the chosen majorant is unavailable, not that the underlying cosine approximation necessarily fails.

**7.4 Explain the factor two.** It comes from bounding the full-space and projected remainders, each under the same spectral enclosure. Explicitly state ||V^T Dj V||_2<=||Dj||_2. Show the add-and-subtract identity used to combine response error, derivative error, and loss residual.

**7.5 Simplify the scalar dependence.** All component bounds are nonnegative multiples of d_j in this formulation. Thus eta_j=d_j h_(m,r), and ||eta||_2=||d||_2 h_(m,r). State this if retained; it clarifies the ablation table's additive component norms and can avoid redundant per-parameter certificate arithmetic. It does not eliminate the cost of computing the gradient components themselves.

**7.6 Consider a directly available tighter residual factor.** Both compressed derivative approximations have right-singular support W_r. Their inner product with the residual depends only on the retained–retained residual. Consequently the projected-residual factor can use the r×r core residual against Y_rr, and a preflight version can use c_r+||Y_rr||_F instead of c_r+||Y||_F. Prove this explicitly and benchmark it as an additional refinement; do not silently change the formula while retaining old experimental numbers.

**7.7 Preserve the distinction between a priori and evaluated bounds.** A plan based on actual residuals requires a projected calculation. Give separate algorithm names, inputs, and costs. A evaluated-residual bound can strengthen a final decision but cannot repair an invalid Krylov span.

**7.8 Quantify conservatism.** The primary table's bounds divided by reference discrepancies are approximately 5.7e4, 3.1e6, 1.3e5, and 2.0e5. Report this gap and its decomposition. The very small response contribution indicates derivative-tail and compression terms dominate these cases. Explain which term changes the selected rank or depth rather than describing all tail improvements equally.

**7.9 Fix the Taylor implementation's coupled infeasibility behavior.** `cosine_tails` returns `(inf,inf)` when the derivative ratio is invalid even if the value ratio was valid. The Chebyshev implementation handles the two tails separately. For example z=3 and degree=1 yield q=2, a value ratio 9/12<1 but derivative ratio 9/6>1. Return the finite value bound and infinite derivative bound independently, and clarify the Appendix A wording. This affects the legacy ablation's efficiency rather than making an overestimated bound unsafe.

**7.10 Avoid overstating upward conversion.** Eighty-digit Decimal arithmetic followed by one upward binary64 step is not a documented interval proof of every preceding operation, particularly under extreme cancellation in 1-ratio. Either prove the rounding enclosure over the supported domain or label it as a conservative high-precision numerical evaluation. Keep the existing warning that the full algorithm is not interval-certified.

**7.11 Add tail-domain tests.** Cover t=0, rho=0, negative t, parity boundaries, ratios just below/at/above one, large z, underflow/overflow, and comparison with high-precision direct series. For no feasible depth, provide an informative reason and safe error return.

**8. Selection, optimization, and computational cost — main.tex:301–364**

**8.1 Add pseudocode.** Algorithm 1 should take B, Y, times, supplied norm bounds, epsilon, and the candidate grid; compute one thin SVD; scan executable ranks and depths; choose the lexicographic minimum (mr,r,m); and return either a validated plan or a diagnostic failure. Algorithm 2 should construct the basis, verify the relevant representation conditions, assemble the inexact gradient, and return clear status information. The current equations alone are insufficient to reproduce execution details.

**8.2 Specify zero/deficient-block behavior and rank admissibility.** The selector currently considers SVD ranks that the basis constructor can reject. Filter admissible ranks consistently or provide a documented fallback. Nonzero is different from numerically resolvable under the specified threshold.

**8.3 Validate cached-plan identity.** The cache is justified only while B, targets, times, directions/norm bounds, parameter box, compression method, and tail method are unchanged. The current public API accepts a raw plan and checks its old total bound against the new tolerance, without establishing that it belongs to the current inputs. Use a typed immutable plan with input identity metadata or explicitly restrict and validate reuse. Mutating a NumPy array inside a frozen dataclass does not make the underlying data immutable.

**8.4 Do not silently truncate noninteger depths.** `preflight_gradient` converts supplied depths to int before passing them to a stricter validator. Reject fractional values rather than accepting 4.9 as 4. Make the minimum depth consistent across APIs and the theorem.

**8.5 Add the projected-stationarity corollary.** For the convex box C, define G(theta)=theta-Pi_C(theta-g) and Ghat(theta)=theta-Pi_C(theta-ghat). Nonexpansiveness gives ||G-Ghat||<=epsilon, hence ||G||<=||Ghat||+epsilon. This justifies the learning stopping test. Specify the unit step and parameter scaling; it is a projected-gradient mapping norm, not a parameter-recovery certificate.

**8.6 Write the actual Gauss–Newton step.** Define the approximate J^T J/s matrix, damping max(1e-10,1e-8 trace(H)), projected trial step, fallback, initial certificate stage, tightening formula, minimum stage, maximum iterations, and backtracking factor. These values are in code but not all in the manuscript. State the larger learning grid 2,4,...,40; it differs from the other grids.

**8.7 Distinguish direction certification from an exact Armijo test.** The code uses u=ghat^T d+epsilon||d||, and accepts using Phi_trial<=Phi_current+c alpha u. A negative u guarantees a certified descent direction and, with exact losses, a decrease for an accepted step. Because u is an upper bound on g^T d, this is not the same condition as classical Armijo using the exact derivative. State the implemented condition and what is proved. For complete certification, derive loss enclosures and an acceptance inequality that accounts for their errors; otherwise retain the explicit floating-point qualification.

**8.8 Connect certification to optimizer behavior conservatively.** A gradient bound does not certify the approximate Hessian or global convergence. A feasible-direction check can validate a step produced by that Hessian, which is exactly the appropriate claim. Give failure exits and make clear that stopping, local optimality, identifiability, and recovery are separate outcomes.

**8.9 Broaden the resource accounting.** Keep mr as a nominal count, but report Hamiltonian and direction actions, block-call widths, orthogonalization, small dense algebra, plan/SVD time, loss acceptance, and auditing separately. The actual work under deflation is sum_i r_i; optimizing mr is not optimizing that realized count. If A is assembled from H0 and p directions, one A-action itself contains additional direction work unless supplied as a fused operator.

**8.10 Add a calibrated cost-model ablation.** Consider selection using measured operator/block costs plus n k^2 and k^3 terms, and compare its choices with mr selection. Treat the calibration cost explicitly and test out of sample. The present method can have fewer operator vectors and still cost more, so improving the objective is more useful than polishing a nominal-work headline.

**8.11 Avoid avoidable width-path storage.** `probe_compression_path` materializes coordinate arrays for every rank after one SVD. The sum of their sizes scales as O(n b^2), although only one rank is eventually needed. Use singular-value prefix/suffix statistics and lazy selected coordinates if b is to scale. Include SVD time/storage and target-block transformation costs in the complexity discussion. The manuscript's O(nmr) statement is explicitly about basis storage, not all memory, and should remain so.

**8.12 Use comparable gradient assembly.** The projected-adjoint contraction is an efficient way to compute the same fixed-projector gradient. Compare joint/full selection with the same assembly backend when only gradients are needed. Gauss–Newton needs derivative blocks, so explain why that use case differs. Otherwise the benchmark mixes rank-selection effects with implementation choices.

**9. Experimental protocol and interpretation — main.tex:366–680**

**9.1 Preserve provenance but stop treating it as the main scientific achievement.** SHA-256 verification is valuable for reproducibility. It does not make synthetic Gaussian index probes a physical electronic-structure measurement. Use “SuiteSparse PARSEC matrices with constructed probes and parameter directions” in major claims. Keep physical inference results separate from operator benchmarks.

**9.2 Put complete benchmark construction in Methods.** State field centers and widths, probe centers/offsets/widths, the phase modulation, column normalization, seeds where applicable, matrix scaling constants, target/evaluation parameters, times, depth grids, and all tolerances. Some are only recoverable from `model.py`. Provide a table of B singular values or their decay and the numerical rank convention.

**9.3 Remove prospective implications from “confirmation.”** The text correctly disclaims preregistration. Rename these roles as historical development/held-out inputs or put the history in Methods. A revised method evaluated repeatedly on the same matrices has not acquired prospective confirmation merely by preserving a label.

**9.4 Equalize comparator opportunities.** The scalar polarization grid is smaller. Either use comparable grids and stopping criteria or mark the comparison as limited. Always separate published algorithms, faithful reproductions, and author implementations. Do not compare a certified method against an oracle without immediately explaining the oracle's unavailable information and excluded costs.

**9.5 Retain executable online stopping.** This is a strong addition to the study. Report observed failures as well as successes over difficult spectra, longer times, and tight tolerances. Two successive small gradient changes can plateau while bias remains. The fixed-rank heuristic at tighter tolerances is a useful negative control, but it is not an adaptive-compression competitor.

**9.6 State the absolute tolerance's scale.** Gradient norms of approximately 0.00326–0.0135 mean 1e-3 is about 7.4%–30.7% of ||g|| in the primary table. This is not a defect, but readers need the relative scale. The much smaller achieved errors mostly reveal conservatism rather than a demonstrated need for that precision.

**9.7 Report the correct end-to-end work savings.** From Table 10's stored medians, the loose-target Krylov saving is (504-396)/504=21.4%, while total Hamiltonian work saving is (2460-2352)/2460=4.39%. At the strict target Krylov work rises by 17.9% and total work by 2.92%. State these numbers beside runtime ratios. They make the limitation clearer than “remove the work benefit.”

**9.8 Explain dependence among the 288 runs.** There are 2 sizes × 3 truths × 3 noise settings = 18 size/truth/noise datasets, then 2 starts × 2 stopping targets × 4 methods. Only 12 of the 18 datasets are noisy. Noise draws are reused across methods/starts/targets, appropriately for paired comparisons; the same seed construction also couples sizes. Do not interpret 288 as independent experimental replications or attach a naive binomial confidence interval to 288 successes.

**9.9 Separate noise and clean recovery plots.** At fixed noise, more optimization accuracy need not reduce error to the generating truth. Show loss, projected-stationarity residual, held-out prediction error, and parameter error separately. Annotate clean/noisy cases, truth, and start. Explain whether reported times are repeated timings or single timings of different cases.

**9.10 Define the actual noise law.** The larger-study code uses N=sigma(Z+Z^T)/2 with independent standard-normal entries. Diagonal variance is sigma^2 and off-diagonal variance sigma^2/2; paired symmetric entries are dependent. State this, the seed rule, and independence across observation times/realizations. It also helps explain the Frobenius weighting of duplicated off-diagonal entries. Do not call sigma the identical standard deviation of every entry.

**9.11 Expand noisy experiments for a high-impact submission.** Vary the amplitude-noise level and test physically motivated shot/readout noise only after specifying a measurement protocol. Add model mismatch, different truths, broader starts, boundary truths, correlated noise, and observation budgets. Define success and statistical summaries before running the additional tests. These are proposed experiments, not implied requirements for every numerical-analysis paper.

**9.12 Diagnose the changed measurement design.** The successful larger problem changes the identity shift, probes/carriers, times, field strengths/widths, system size, and optimizer. Add controlled ablations holding all but one factor fixed, or explicitly treat the two studies as separate examples. The current caveat is correct and should be retained. Do not infer that sharper certificates caused the change from 50% to 100% recovery when all methods share those outcomes.

**9.13 Improve the original-failure evidence.** Label every failed fit with truth and start. Report objective at the truth and fit, active constraints, independent projected residual, and Hessian finite-difference step sensitivity. A positive numerical Hessian at an approximately stationary point is evidence of a nonglobal local minimum, not a rigorous existence proof. For boundary fits, check active-set KKT signs and feasible curvature if making stronger local-minimum claims.

**9.14 Make identifiability information visible.** Include the larger-study Jacobian singular values/condition numbers already recorded, not only the old two-truth numbers. Define whether the Jacobian vectorizes all symmetric entries or uses a weighted upper triangle. Local full rank does not prove uniqueness over the box. Show the effect of carriers and times on sensitivity as part of the measurement-design ablation.

**9.15 Preserve the cosine ambiguity discussion.** A and -A give the same cosine. The restricted box excludes that particular ambiguity here but not all aliases. Multiple times can introduce additional identifiability questions. If uniqueness is not proved, retain the current limited claim and report systematic multistart searches as evidence rather than proof.

**9.16 Strengthen scale evidence.** The largest operator benchmark has n=19,896 but p=3 and b=8. The learning sizes are 256 and 1024 with p=3. Stress matrices are only order 80, and 17/36 reach the full dimension. Add regimes with increasing n, p, b, horizon, and spectral spread; keep reduced dimension well below n in a genuine scalability experiment. Report memory and both sparse-operator and dense-reduction costs. Full-space cases validate consistency, not scalable efficiency.

**9.17 Strengthen the stress design.** The 36 cells form a useful factorial grid but apparently do not constitute many random replicates per cell. Add multiple seeds, adversarial partial deflation, clustered spectra, noncommuting directions, weak/flat probe spectra, and longer times. Show failures/infeasible plans and their reasons. A coverage fraction on 36 selected cases is not a theorem about all floating-point executions.

**9.18 Establish the reference accuracy floor.** The sparse augmented exponential is a meaningfully independent construction, but not exact arithmetic. Add high-precision dense checks on small difficult matrices and a reference-refinement study. Report a numerical uncertainty estimate or agreement threshold where appropriate. Values near 1e-15 should be labeled numerical agreement, as the current Figure 2 caption already does.

**9.19 Check derivative references beyond shared kernels.** Some small tests use the same cosine/Loewner helper as the implementation. Preserve the independent exponential-Fréchet tests and add directional finite-difference step sweeps of an independently evaluated original loss. A single step 1e-5 is not a universal validation of every derivative scale. Distinguish finite differences of the original loss from the rebuilt projected surrogate.

**9.20 Improve timing design.** Use repeated paired, balanced/rotated orders for all headline comparisons, not only the nine same-depth repetitions. The larger learning loop already rotates method order; say so explicitly. Use a timer protocol suitable for 0.01–0.1 second runs, warm-up, actual backend/thread metadata, and raw paired ratios. A ratio of medians and median of paired ratios differ; choose and label the statistic. Repeated timings and variability across different inverse problems answer different questions.

**9.21 Add uncertainty without pseudoreplication.** Provide timing quantiles or paired uncertainty summaries where sufficient repetitions exist. Aggregate learning performance at the dataset level, preserving paired method structure. Two noise realizations per truth are insufficient for strong noise-robustness claims. Do not manufacture statistical significance from repeated timing measurements on the same four matrices.

**9.22 Show practical value of certification.** The decisive additional experiment is a difficult, application-motivated regime where an uncertified rule returns an unacceptable gradient or stopping decision and the certificate prevents it at an acceptable cost. An expensive-operator regime may also justify mr savings, but do not create artificial wins merely by adding delays. State why a deterministic preflight budget matters in the application.

**10. Every table and figure**

| Item | Required or recommended revision |
|---|---|
| Table 1, inputs | Add a label/cross-reference; define nnz; clarify historical roles; give raw-to-scaled factors and indicate constructed B,D. Six rho decimals need a reason if all cases share essentially 0.98. |
| Table 2, joint selection | Add label; define work units and direction of time ratio in the heading; add absolute runtimes or link directly to them; standardize scientific notation. Include realized dimension/deflation status where relevant. |
| Table 3, four-way ablation | Keep this central evidence. Spell out the method names cleanly; add visual group spacing; increase the scriptsize text. Explain that component norms add here because they are proportional to the same d-vector, not because norms generally add. |
| Table 4, matched accuracy | Distinguish observed oracle from executable method in the heading; state denominator of time ratio; retain gradient norm; include achieved discrepancy and depth for the chosen oracle or supply a linked supplementary table. |
| Table 5, rank-first policy | Compact appendix candidate: all but one cell agree. Explain this limited empirical difference instead of implying a large selection-policy advantage. Define the compression-budget constraint. |
| Table 6, incremental adjoint | Group by tolerance or matrix consistently; show every stopping checkpoint or provide a source table; label the last column as unavailable post hoc information. Add time variability and define the assembly backend. |
| Table 7, original recovery | Report truth/start identities in detailed data; keep failures; separate cross-case medians from timing repetition summaries. Define parameter error explicitly as an Euclidean norm. |
| Table 8, failed fits | Add truth/start columns; mark active bounds; show Hessian-step sensitivity and exact-loss comparison at truth; keep “numerical evidence” wording. |
| Table 9, larger learning | Too dense at current scriptsize. Split recovery/accuracy from resource use, or move full factorial rows to an appendix. Distinguish 18 runs from independent datasets; include parameter-error quantiles/maxima and independent stationarity. |
| Table 10, learning costs | Add explicit percentage changes, direction-action costs, and identical ratio orientation to other tables; show whether ratios use medians or paired cases. |
| Figure 1, bound/discrepancy | Keep; add compression-only bias versus projection error if available, actual r, panel letters, and a bound/error ratio or annotation. Increase fonts. The plateau needs explanation. |
| Figure 2, runtime/error | Label sampled depths, distinguish online and oracle points, show timing variation, identify the reference floor, and explain connecting lines are sampled sequences rather than continuous guaranteed frontiers. |
| Figure 3, runtimes | Avoid calling four chemically different matrices a controlled scaling law. State sizes or nnz, explain log scale, and keep the excluded reference/oracle costs explicit. |
| Figure 4, rank/depth map | Add missing label; use a discrete rank color scale and readable integer ticks; add realized work/feasibility and a meaningful no-feasible-plan state. Expand beyond Si2 if drawing broad adaptation conclusions. |
| Figure 5, learning | Rename the file `probe_spectrum.pdf` to reflect learning trajectories; there is no probe spectrum in this image. Mark clean/noisy final points and starts; add stationarity plots; clarify representative trajectory selection. |

The current PDF uses mixed full/proposed and proposed/full ratios. Standardize throughout, preferably proposed/baseline with values below one meaning a reduction. Do not let a caption convention change reverse the interpretation from one table to another.

**11. Page-by-page visual and typesetting review**

All 15 pages were rendered and inspected. I did not see gross clipping, overlapping text, unresolved `??` references, or missing figures in this snapshot. The main issues are density, navigation, and float placement rather than a broken PDF.

| PDF page | Specific revision |
|---|---|
| 1 | Dense abstract and introduction; reduce the abstract and make the main practical claim easier to identify. The all-caps title is legible but generic. Check affiliation/email accuracy; do not infer that an external email is an error. |
| 2 | Introduce g^(r) explicitly; give the block cancellation identity. A notation table elsewhere would reduce the symbol load. |
| 3 | Theorem 3 and tail formulas are readable; add the explicit derivative counterexample and make the exact-span requirement prominent. |
| 4 | Separate algorithm selection from optimizer implications; add pseudocode and projected-stationarity result. |
| 5 | Methods text is compressed; put all settings in a structured table or appendix, and simplify the repeated caveats. |
| 6 | Table 3 is small; Table 2 precedes its subsection heading; floats interrupt the discussion. Add table labels and control placement without forcing every float here. |
| 7 | Figure 1 is useful but labels are small; the surrounding prose is split by floats. Table 5 can move to an appendix. |
| 8 | Figure 2 needs larger axes/legends and panel letters; Table 6 needs improved grouping and numerical alignment. |
| 9 | Large gaps between Figures 3 and 4; combine/rebalance the float layout rather than shrinking text. The rank colorbar should be discrete. |
| 10 | Failure study is scientifically useful; label fit origins and improve Table 8 heading widths. Keep the local-minimum qualifier. |
| 11 | Table 9 is especially dense. Split it or move detailed rows while preserving a concise main-text learning summary. |
| 12 | Figure 5's many outcomes need noise/start encoding; conclusion begins beneath a large figure and should not feel like another caption. |
| 13 | Appendix formulas and references share a dense page. Separate proof material and reference flow cleanly; shorten the availability paragraph's implementation detail. |
| 14 | Bibliography uses inconsistent completeness and uncited entries; see the audit below. |
| 15 | Only two references occupy the final page. Both are currently uncited. Removing irrelevant entries and letting TeX reflow should eliminate this near-empty page without font compression. |

Use a journal-supported class once the venue is chosen rather than maintaining a bespoke imitation. Add keywords and MSC codes if the selected mathematics venue expects them. The current style defines such environments but the manuscript does not use them. Use upright units, `\times10^{...}`/consistent scientific-number formatting, aligned numeric columns, consistent “Fréchet,” and consistent “rank–depth.” Define every symbol at first use, label every table/figure that matters, and cite each before or near appearance. Define a PDF title/author in hyperref metadata. The custom title command currently does not display `\date`, despite the source setting one; decide intentionally whether the preprint should show a revision date.

**12. Bibliography and external-model audit**

Seven bibliography entries are currently uncited: `lanczos1950`, `golub2010`, `beckermann2018`, `higham2014`, `rubensson2024`, `cramer2010`, and `childs2017`. Cite the ones that support a specific argument and remove the rest. Do not retain unrelated quantum references to make a numerical paper look more physical.

| Entry | Action |
|---|---|
| DLMF | Cite both expansion and inequality with exact equation links; retain access/version information consistently. |
| Higham 2008 | Keep as the general matrix-function reference; verify publisher/DOI formatting in the chosen bibliography style. |
| Saad 1992 | DOI/title/volume/pages matched the retrieved DOI metadata. Keep. |
| Hochbruck–Lubich 1997 | DOI/title/volume/pages matched retrieved metadata. Keep. |
| Lanczos 1950 | Currently uncited; cite for a specific historical statement or remove; complete page range/DOI if retained. |
| Golub–Meurant 2010 | Currently uncited; potentially relevant for quadrature/moment exactness if explicitly discussed. |
| Gutknecht 2007 | Keep for block Krylov background; verify chapter details and give a stable access link if available. |
| Xu–Chen | Current text correctly gives the published 2025 volume/pages although the citation key says 2022. DOI metadata also records earlier online publication; distinguish online and issue year rather than “correcting” to the key. |
| Kandolf–Relton 2017 | Keep; discuss how action-of-Fréchet approaches differ from a sandwiched loss-gradient bound. Complete DOI check before submission. |
| Kandolf et al. 2021 | Add DOI 10.1002/nla.2401 after verifying final publisher metadata; compare its low-rank object with compression of B. |
| Kressner 2019 | Explain the relation to bivariate matrix functions; complete chapter DOI and metadata check. |
| Singh–Barros–Li 2026 | Verified title/authors/arXiv identity. Expand the technical comparison and pin the preprint version used. |
| Al-Mohy–Higham 2009 | Keep for dense exponential Fréchet computation; it is not the complete citation for SciPy's sparse exponential-action algorithm. |
| Davis–Hu 2011 | Clarify that “1” is an article number if using that style; include DOI and precise matrix collection links. |
| Chelikowsky et al. 1994 | Retain for provenance/physics background; do not use it to imply that index-Gaussian probes are physical orbitals. |
| Chen et al. 2022 | DOI/title/volume/pages matched retrieved metadata. Explain what spectral/residual information its bounds need. |
| Simunec | Update the arXiv-only reference to the published work, DOI 10.1002/nla.2571, using final metadata. The author's deposited published paper identifies acceptance in May 2024. |
| Beckermann et al. 2018 | Currently uncited; DOI/title/pages matched. Low-rank updates of A and low-rank compression of B are different operations; cite only with an explicit connection. |
| Higham–Relton 2014 | Currently uncited; retain only if higher derivatives/conditioning enter the argument. |
| Rubensson 2024 | Currently uncited; DOI/title/pages matched. Same relevance test as the preceding entry. |
| Van Loan 1978 | Complete page range/DOI and explain the augmented-exponential block identity. |
| Shabani et al. 2011 | Keep only for a specific measurement-learning comparison; compressive sensing does not automatically validate the present observation model. |
| Wiebe et al. 2014 | State the distinct access model and resources relative to this classical implementation. |
| Bairey et al. 2019 | Contrast local measurements with coherent amplitude-block observations. |
| Cramer et al. 2010 | Currently uncited; remove unless a substantive tomography connection is developed. |
| Childs et al. 2018 | Currently uncited; quantum speedup is expressly not claimed, so likely remove. |

Add the actual sparse matrix-exponential action algorithm/software references used for `expm_multiply`, and the relevant optimizer references if the optimization implementation is discussed as a substantive part of the method. Verify every DOI, author list, volume, issue/article number, page range, and publication status in a final bibliography pass. The DOI check in this review was partial: some requests were rate-limited. No claim is made that all 26 entries have been externally validated.

The published Simunec information is available in the [author's institutional deposit](https://ricerca.sns.it/retrieve/297e1658-d44e-4e4d-9fd9-c60cc025895a/Numerical%20Linear%20Algebra%20App%20-%202024%20-%20Simunec%20-%20Error%20bounds%20for%20the%20approximation%20of%20matrix%20functions%20with%20rational%20Krylov.pdf).

**13. How to use the requested presentation models**

The two arXiv examples are different kinds of papers. [2301.05707](https://arxiv.org/abs/2301.05707), “Machine Learning Assisted Vector Atomic Magnetometry,” presents an experimental sensing application. Borrow its problem-to-measurement-to-validation narrative, not implied experimental authority. [2607.06758](https://arxiv.org/pdf/2607.06758), “Adaptive, Matrix-Free Low-Rank Approximation,” is much closer to the numerical-method direction: explicit access assumptions, algorithms, guarantees, and empirical tradeoffs. Neither preprint's formatting is evidence of acceptance by Nature, and their scientific claims should not be adopted as validated solely because they are style models.

For a SISC-oriented revision, use this sequence: introduction and nearest work; objective/access assumptions; loss-aware compression; separate polynomial exactness and tails; robust selection/evaluation algorithms; optimization implications; controlled numerical evidence; limitations. Keep the main proof chain visible and move exhaustive parameter grids and historical failure diagnostics to appendices while summarizing them honestly in the main text.

For Nature Computational Science, restructure around Results, Discussion, and Methods, with the essential theorem explained accessibly and detailed proofs in supplementary material. Its current Article instructions specify up to 3,500 main-text words, a 150-word abstract, and six main display items. The present ten tables plus five figures need selection and consolidation, not simply smaller typography. [Official content-type instructions](https://www.nature.com/natcomputsci/content). A possible six-item plan is: method schematic; compression-scaling evidence; certified/observed work frontier; application-level learning result; robustness/measurement-design figure; concise method-comparison table. This restructuring is worthwhile only after the scientific case is strong enough.

The [Cartan synthesizer repository](https://github.com/kemperlab/cartan-quantum-synthesizer) is useful as a model for introducing the software's core objects and walking through an executable physical example. Your README currently explains reproduction and limitations better than it explains how a new user solves one small problem. Borrow the quick-start narrative, while retaining your stronger provenance and validation records. Do not copy another repository's wording or assume its older packaging is a current release standard.

**14. Repository, arXiv package, and research provenance**

**14.1 Add a small executable quick start.** Show how to construct an affine symmetric family, specify B and observations, select a plan, evaluate a gradient, inspect the theoretical-bound scope/status, and handle infeasibility. Supply expected dimensions and a compact output. The demonstration should not require the full external matrix download.

**14.2 Document the public API.** Explain the three useful layers: model/operator access, plan/certificate construction, and evaluation. State what is cached, what is copied, shape conventions, sparse direction support, real-symmetric restrictions, and failure returns. Update package descriptions that still suggest broad “quantum operator learning” without the cosine qualification.

**14.3 Add clean-environment CI.** Test declared Python versions with a sensible supported dependency matrix, and include installation from built wheels/source distributions. Run the independent deflation regressions and selected numerical smoke tests. The review's passing Python 3.13 run does not verify the full 3.10–3.13 support range.

**14.4 Provide a reproducibility environment distinct from broad compatibility ranges.** `numpy>=1.26,<3` is a compatibility constraint, not a locked recreation of the reported experiment. Supply an exact environment/lock specification, platform/backend details, and commands for approximate cross-platform numerical agreement. Exact rounded digests may differ without changing scientific conclusions; retain both integrity and tolerance-based checks.

**14.5 Add a documented manuscript build command.** State engine, required TeX packages, generated-number dependency, and build invocation. Build in a clean directory and check warnings, missing references, included graphics, and PDF text. Do not imply `main.tex` alone is standalone: it needs the custom style, `numbers.tex`, and five figure PDFs. Prepare an arXiv source bundle containing all dependencies, or embed generated macros if a truly self-contained source is desired.

**14.6 Make the validator test scientific consistency.** It currently checks hashes, files, counts, and stored conditions, but passing it did not detect the deflation defect. Add regenerated-table consistency, independent numerical checks, and the new regressions. It should not use a hard-coded count of exactly five figures as a proxy for manuscript validity. Byte equality between duplicate generated files does not establish that their values follow from raw data.

**14.7 Remove the blanket wording blacklist.** `validate_release.py` rejects “AI,” “artificial intelligence,” “ChatGPT,” and “agentic” anywhere in the paper/README/site. That is not a scientific validity check and could reject legitimate references or an accurate disclosure. Remove it or replace it with a narrowly justified content check. Any disclosure required by the selected journal must remain possible; do not conceal applicable assistance or invent declarations.

**14.8 Add release identity and citation metadata.** Use a tagged release/commit, `CITATION.cff`, a stable version, and an archival DOI if deposited. Link the paper to the exact artifact version. Keep both the full source digest and numerical digest in machine-readable data; the main paper need not foreground digest prefixes over science.

**14.9 Preserve raw and derived data clearly.** Provide machine-readable source data for every figure/table, a mapping from records to display items, and the generator commands. Keep one authoritative result source and verify packaged copies automatically. Preserve original negative cases and label pilot-informed additions honestly.

**14.10 Audit the public package independently of the review directory.** The working tree already contained many modifications before this review. Establish the intended release commit and avoid overwriting unrelated work. Review notes, counterexamples under investigation, temporary renders, and unpublished editorial correspondence should not be accidentally included in the public manuscript package. A finalized regression counterexample belongs in tests once repaired.

**14.11 Finish submission-specific front matter accurately.** Confirm the author's affiliation, corresponding address, contributions, funding/acknowledgments, and any journal-required declarations with the author. Do not fabricate them. Separate data availability from code availability if the chosen venue requires that format; retain clear external-matrix terms and download provenance.

**15. Priority, effort, and completion criteria**

These are working estimates, not promises of new positive scientific results. They assume the present compact implementation and access to the current data. Full experiment runtime, independent external review, and any physical-data acquisition are additional uncertainties.

| Work package | Estimated focused effort | Completion criterion |
|---|---:|---|
| Deflation, returned status, and plan safety | 2–4 days | Supplied and extended counterexamples pass; actual span assumptions/status are enforced. |
| Theory clarification and new corollaries | 2–4 days | Compressed objective, cancellation identity, noisy-target and projected-stationarity results are proved and aligned with code. |
| Tail evaluation and numerical edge cases | 1–3 days | Independent tail tests pass; rounding scope and infeasibility behavior are explicit. |
| Cost accounting and matched assembly | 2–4 days | Comparable backends and complete timing/action ledgers; no interpretation-changing omitted costs. |
| Reproduction after repairs | 1–3 days plus measured compute time | Full outputs regenerated, audited, and compared against old conclusions. |
| Stronger noise/design/scaling experiments | 1–3 weeks | Predeclared added cases answer specific novelty/utility questions, including failures. |
| Introduction and abstract | 1–2 days | Claims match evidence and nearest-work distinction is explicit. |
| Methods/optimization reproducibility | 1–2 days | All implementation settings recoverable without searching code. |
| Figures, tables, layout | 2–3 days | Legible final-size displays, consistent ratios, clean 15-page-or-restructured flow. |
| Bibliography and repository/arXiv packaging | 2–3 days | Complete source bundle, checked citations, tested installation/build, traceable release. |

A careful specialist-journal revision is plausibly several weeks, roughly four to seven weeks if the expanded experiments are included and no new major defect appears. A Nature Computational Science campaign is a research project rather than a copy-edit: allow months, with no guarantee that the required scientific advance will emerge.

Submit only after these conditions hold: the deflation defect is fixed; theoretical assumptions and implementation status agree; every headline is recomputed from the repaired code; the comparison uses appropriate baselines and matched tasks; failures and uncertainty remain visible; the originality relative to nearest work is specific; and the package installs, tests, rebuilds, and reproduces in a clean supported environment. The current paper should be developed around its defensible error-control contribution, while any higher-impact claim should follow from new evidence rather than presentation alone.
