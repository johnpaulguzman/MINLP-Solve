 ("ALPHA [6,1] COMPUTATION")
            SPR = (y-12.0+28.0)/(1+0.0) - y;  B = Max[ (28.0)/(1+0.0) - z, (z-16.0+28.0)/(1+0.0) - z ];  A = Min[ (z+x)*(1+0.0) - 28.0 + 14.0, ((z+x)*(1+0.0)-28.0+30.0)/(1+0.0) - z, ((z+x)*(1+0.0)-28.0+26.0)/(1+0.0) - x, ((z+x)*(1+0.0)-28.0+42.0)/(1+0.0) - z - x ];  mu = { { 15.0, 17.0, 13.0 } };  sig = { { 1.0, 0.0, 0.0 }, { 0.0, 4.0, 0.0 }, { 0.0, 0.0, 0.25 } };
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {z, SPR, intlimit}, {x, B, intlimit}, {y, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [6,1]")
            