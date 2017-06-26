 ("ALPHA [7,6] COMPUTATION")
            SPR = ((y+z)*(1+0.0)-28.0+42.0)/(1+0.0) - y - z;  B = Max[ (z-12.0+42.0)/(1+0.0) - x - z, ((x+z)*(1+0.0)-26.0+42.0)/(1+0.0) - x - z ];  A = Max[ (42.0)/(1+0.0) - x - y, (x-14.0+42.0)/(1+0.0) - x - y, (y-16.0+42.0)/(1+0.0) - x - y, ((x+y)*(1+0.0)-30.0+42.0)/(1+0.0) - x - y ];  mu = { { 15.0, 17.0, 13.0 } };  sig = { { 1.0, 0.0, 0.0 }, { 0.0, 4.0, 0.0 }, { 0.0, 0.0, 0.25 } };
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, SPR, intlimit}, {y, B, intlimit}, {z, A, intlimit}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [7,6]")
            