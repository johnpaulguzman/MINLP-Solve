 ("ALPHA [6,2] COMPUTATION")
            SPR = (z-12.0+28.0)/(1+0.0) - z;  B = Max[ (28.0)/(1+0.0) - y, (y-16.0+28.0)/(1+0.0) - y ];  A = Min[ (y+z)*(1+0.0) - 28.0 + 14.0, ((y+z)*(1+0.0)-28.0+30.0)/(1+0.0) - y, ((y+z)*(1+0.0)-28.0+26.0)/(1+0.0) - z, ((y+z)*(1+0.0)-28.0+42.0)/(1+0.0) - y - z ];  mu = { { 15.0, 17.0, 13.0 } };  sig = { { 1.0, 0.0, 0.0 }, { 0.0, 4.0, 0.0 }, { 0.0, 0.0, 0.25 } };
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {y, SPR, intlimit}, {z, B, intlimit}, {x, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [6,2]")
            