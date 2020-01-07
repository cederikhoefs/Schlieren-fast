
kernel void schlieren( global bool* schlieren, const double Scale, const int Resolution, const int Iterations, const double vx, const double vy) {

	const int idx = get_global_id(0);

	const int i = idx % Resolution;
	const int j = idx / Resolution;

	double x = ((double)i / Resolution - 0.5) * Scale - vx;
	double y = (0.5 - (double)j / Resolution) * Scale - vy;
	
	double delta = 0.5 * (Scale / Resolution);
	
	double xdx = x + delta;
	double ydx = y;

	double x_dx = x - delta;
	double y_dx = y;

	double xdy = x;
	double ydy = y + delta;

	double x_dy = x;
	double y_dy = y - delta;
	
	for (int k = 0; k < Iterations; k++) {

		if (x_dx * xdx < 0.0 || x_dy * xdy < 0.0) { //VzW
			schlieren[idx] = true;//k;
			return;
		}

		double xdx_new = -xdx - ydx;
		double ydx_new = xdx * ydx;

		double x_dx_new = -x_dx - y_dx;
		double y_dx_new = x_dx * y_dx;
			   
		double xdy_new = -xdy - ydy;
		double ydy_new = xdy * ydy;

		double x_dy_new = -x_dy - y_dy;
		double y_dy_new = x_dy * y_dy;

		xdx = xdx_new;
		ydx = ydx_new;

		x_dx = x_dx_new;
		y_dx = y_dx_new;

		xdy = xdy_new;
		ydy = ydy_new;

		x_dy = x_dy_new;
		y_dy = y_dy_new;
	
	}

	schlieren[idx] = false;

}
/*
kernel void twotothenegativek( global bool* schlieren, const double Scale, const int Resolution, const int Iterations, const double vx, const double vy) {

	const int idx = get_global_id(0);

	const int i = idx % Resolution;
	const int j = idx / Resolution;

	double x = ((double)i / Resolution - 0.5) * Scale - vx;
	double y = (0.5 - (double)j / Resolution) * Scale - vy;
	
	double delta = 0.5  * (Scale / Resolution);
		
	for (int k = 0; k < Iterations; k++) {
		
		if(sign((x - delta)-(1.0 - pow(2.0, -(double)k))) != sign(((x + delta)-(1.0 - pow(2.0, -(double)k))))){
			schlieren[idx] = true;	
			return;
		}
	
	}

	schlieren[idx] = false;

}
*/