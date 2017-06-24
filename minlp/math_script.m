 ("ALPHA [2,1] COMPUTATION")
SPR = 16.0;
B = Min[ x - 16.0 + 14.0, (x-16.0+30.0)/(1+0.0) - x ];
A = Min[ x - 16.0 + 12.0, (x-16.0+26.0)/(1+0.0) - y, (x-16.0+28.0)/(1+0.0) - x, (x-16.0+42.0)/(1+0.0) - y - x ];
mu = { { 15.0, 17.0, 13.0 } };
sig = { { 1.0, 0.0, 0.0 }, { 0.0, 4.0, 0.0 }, { 0.0, 0.0, 0.25 } };
h = 3;
r = { {x, y, z} };
intlimit = Infinity;
mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
intres = Integrate[f, {x, SPR, intlimit}, {y, -intlimit, B}, {z, -intlimit, A}];
Print[intres]; (" OUTPUT # 1 ")
