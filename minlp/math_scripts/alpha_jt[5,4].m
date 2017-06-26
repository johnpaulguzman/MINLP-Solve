 ("ALPHA [5,4] COMPUTATION")
            SPR = (z-12.0+26.0)/(1+0.0) - z;  B = Max[ (26.0)/(1+0.0) - x, (x-14.0+26.0)/(1+0.0) - x  ];  A = Min[ (x+z)*(1+0.0) - 26.0 + 16.0, ((x+z)*(1+0.0)-26.0+30.0)/(1+0.0) - x , ((x+z)*(1+0.0)-26.0+28.0)/(1+0.0) - z, ((x+z)*(1+0.0)-26.0+42.0)/(1+0.0) - x - z ];  mu = { { 15.0, 17.0, 13.0 } };  sig = { { 1.0, 0.0, 0.0 }, { 0.0, 4.0, 0.0 }, { 0.0, 0.0, 0.25 } };
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, SPR, intlimit}, {z, B, intlimit}, {y, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [5,4]")
            