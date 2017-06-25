 ("ALPHA [3,1] COMPUTATION")
            SPR = 12.0;  B = Min[ y - 12.0 + 14.0, (y-12.0+26.0)/(1+0.0) - y ];  A = Min[ y - 12.0 + 16.0,  (y-12.0+30.0)/(1+0.0) - z, (y-12.0+28.0)/(1+0.0) - y, (y-12.0+42.0)/(1+0.0) - z - y ];  mu = { { 15.0, 17.0, 13.0 } };  sig = { { 1.0, 0.0, 0.0 }, { 0.0, 4.0, 0.0 }, { 0.0, 0.0, 0.25 } };
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {y, SPR, intlimit}, {z, -intlimit, B}, {x, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [3,1]")
            