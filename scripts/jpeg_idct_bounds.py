"""Worst-case intermediates of libjpeg's accurate integer IDCT (jidctint.c), for the limits in
hip/port/cheshirejpg/jpegTypes.hpp (kMaxDequantized, kMaxColumnPass).

The GPU decoder computes the IDCT in 32-bit integers where libjpeg's C path uses 64-bit JLONG and
its SIMD paths dequantise in 16 bits and pack the column pass to 16 bits with saturation. The three
agree exactly when no intermediate overflows 31 bits and no column-pass output leaves int16. This
script evaluates every sign pattern of 8 inputs at a given magnitude, through one pass of the exact
integer code, and prints the largest intermediate and output: pass 1 with the dequantised
coefficients as input, pass 2 with the column-pass outputs.

    python3 scripts/jpeg_idct_bounds.py
"""
import itertools
C=dict(F0298=2446,F0390=3196,F0541=4433,F0765=6270,F0899=7373,F1175=9633,F1501=12299,F1847=15137,F1961=16069,F2053=16819,F2562=20995,F3072=25172)
def d(x,n): return (x+(1<<(n-1)))>>n
def onepass(v, shift_out, track):
    # v: 8 inputs in natural index order 0..7 (already dequantised / workspace)
    z2,z3=v[2],v[6]; z1=(z2+z3)*C['F0541']; tmp2=z1-z3*C['F1847']; tmp3=z1+z2*C['F0765']
    track(z1,tmp2,tmp3)
    z2,z3=v[0],v[4]; tmp0=(z2+z3)<<13; tmp1=(z2-z3)<<13
    t10=tmp0+tmp3; t13=tmp0-tmp3; t11=tmp1+tmp2; t12=tmp1-tmp2
    track(tmp0,tmp1,t10,t11,t12,t13)
    tmp0,tmp1,tmp2,tmp3=v[7],v[5],v[3],v[1]
    z1=tmp0+tmp3; z2=tmp1+tmp2; z3=tmp0+tmp2; z4=tmp1+tmp3; z5=(z3+z4)*C['F1175']
    track(z5)
    tmp0*=C['F0298']; tmp1*=C['F2053']; tmp2*=C['F3072']; tmp3*=C['F1501']
    z1*=-C['F0899']; z2*=-C['F2562']; z3*=-C['F1961']; z4*=-C['F0390']
    track(tmp0,tmp1,tmp2,tmp3,z1,z2,z3,z4)
    z3+=z5; z4+=z5; tmp0+=z1+z3; tmp1+=z2+z4; tmp2+=z2+z3; tmp3+=z1+z4
    track(z3,z4,tmp0,tmp1,tmp2,tmp3)
    outs=[t10+tmp3,t10-tmp3,t11+tmp2,t11-tmp2,t12+tmp1,t12-tmp1,t13+tmp0,t13-tmp0]
    track(*outs)
    return [d(o,shift_out) for o in outs]
def worst(B, shift_out):
    m=[0]; mo=[0]
    def tr(*xs):
        for x in xs: m[0]=max(m[0],abs(x))
    for signs in itertools.product((-1,0,1),repeat=8):
        v=[s*B for s in signs]
        o=onepass(v,shift_out,tr)
        mo[0]=max(mo[0],max(abs(x) for x in o))
    return m[0],mo[0]

if __name__ == "__main__":
    for B in (1024, 2048, 4096, 8191):
        i1, o1 = worst(B, 11)  # pass 1: descale by CONST_BITS - PASS1_BITS
        print(f"pass 1, |input| <= {B:5}: max intermediate {i1:>11} ({i1 / 2**31:.3f} of 2^31), "
              f"max output {o1} ({'fits' if o1 < 32768 else 'exceeds'} int16)")
    for M in (16383, 30607, 32767):
        i2, _ = worst(M, 18)  # pass 2: descale by CONST_BITS + PASS1_BITS + 3
        print(f"pass 2, |input| <= {M:5}: max intermediate {i2:>11} ({i2 / 2**31:.3f} of 2^31)")
