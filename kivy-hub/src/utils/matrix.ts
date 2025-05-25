const M = new Float32Array([   1.8037489,    -0.19094944 ,-306.1731943
      -0.03725462   , 1.73408325, -156.93731623,
       0.0000025 ,   -0.00015512 ,   1.        ]);

const width = 1920;
const height = 1200;

export function transformCoordinates(x: number, y: number) {
  // Convert point to homogeneous coordinates [x, y, 1]
  const point = [x, y, 1];

  // Apply the homography transformation
  // [x']   [M00 M01 M02] [x]
  // [y'] = [M10 M11 M12] [y]
  // [w']   [M20 M21 M22] [1]
  const transformedX = M[0] * point[0] + M[1] * point[1] + M[2] * point[2];
  const transformedY = M[3] * point[0] + M[4] * point[1] + M[5] * point[2];
  const transformedW = M[6] * point[0] + M[7] * point[1] + M[8] * point[2];

  // Convert back from homogeneous coordinates
  let newX = transformedX / transformedW;
  let newY = transformedY / transformedW;

  // Clip coordinates to screen bounds
  newX = Math.min(Math.max(0, newX), width - 1);
  newY = Math.min(Math.max(0, newY), height - 1);

  return [newX, newY];
}
