/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;
using std::uniform_real_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 100;  // TODO: Set the number of particles
	
	// Creates random generator engine
	std::default_random_engine gen;

	// Creates a normal (Gaussian) distribution for x,y, and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle temp_part;
		temp_part.id = i;
		temp_part.weight = 1.0;
		// Sample from normal distributions using:
		// x = dist_x(gen);
		temp_part.x = dist_x(gen);
		temp_part.y = dist_y(gen);
		temp_part.theta = dist_theta(gen);
		
		particles.push_back(temp_part);
		weights.push_back(temp_part.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	
	// Creates random generator engine
	std::default_random_engine gen;

	for (int i = 0; i < num_particles; ++i) {
		// Calculates new x,y, and theta for a given particle
		double x = 0;
		double y = 0;
		double theta = 0;
		// Check for yaw_rate and decide which equations to use
		if (yaw_rate == 0) {
			x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			theta = particles[i].theta;
		}
		else {
			x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			theta = particles[i].theta + yaw_rate*delta_t;
		}
		// Creates a normal (Gaussian) distribution for x,y, and theta
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);
		// Update particles's predicted x,y,theta values with Gaussian noise applied
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

	for (int i = 0; i < observations.size(); ++i) {
		LandmarkObs obs = observations[i];
		double min_dist = std::numeric_limits<double>::max();

		obs.id = -1;

		for (int j = 0; j < predicted.size(); ++j) {
			LandmarkObs pred = predicted[j];
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			if (distance < min_dist) {
				obs.id = pred.id;
				min_dist = distance;
			}
		}
		observations[i].id = obs.id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

	// For all particles
	for (int i = 0; i < num_particles; ++i) {
		// Local variables for particle
		double xp = particles[i].x;
		double yp = particles[i].y;
		double theta = particles[i].theta;
		// Vector of predicted landmarks within sensor range
		vector<LandmarkObs> predicted;

		// For each landmark on the map
		// find which landmarks are within sensor range
		// and add them to predicted vector
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			int lmrk_id = map_landmarks.landmark_list[j].id_i;
			float lmrk_x = map_landmarks.landmark_list[j].x_f;
			float lmrk_y = map_landmarks.landmark_list[j].y_f;

			// Find which landmarks are within sensor range 
			// by checking distance between particle location and landmark location
			if (dist(xp, yp, (double)lmrk_x, (double)lmrk_y) < sensor_range) {
				// Create LandmarkObs variable with the current landmark's data
				LandmarkObs inrange_landmark;
				inrange_landmark.id = lmrk_id;
				inrange_landmark.x = lmrk_x;
				inrange_landmark.y = lmrk_y;
				// Push the landmark id and coordinates to the predicted vector.
				predicted.push_back(inrange_landmark);
			}
		}

		// Convert observations from vehicle coordinates to map coordinates.
		vector<LandmarkObs> observations_map;
		for (int k = 0; k < observations.size(); ++k) {
			LandmarkObs obs_map;
			double xc = observations[k].x;
			double yc = observations[k].y;

			obs_map.id = observations[k].id;
			obs_map.x = xp + (cos(theta) * xc) - (sin(theta) * yc);
			obs_map.y = yp + (sin(theta) * xc) - (cos(theta) * yc);

			observations_map.push_back(obs_map);
		}
		
		// Use dataAssociation() function to associate observations with predicted landmarks.
		dataAssociation(predicted, observations_map);

		// Create associations, sense_x, and sense_y vectors using observations converted to map coordinates.
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		for (int k = 0; k < observations_map.size(); ++k) {
			associations.push_back(observations_map[k].id);
			sense_x.push_back(observations_map[k].x);
			sense_y.push_back(observations_map[k].y);
		}

		// Use SetAssociations() and above vectors to update current particle.
		SetAssociations(particles[i], associations, sense_x, sense_y);
		
		// Calculate weight for each observation using multiv_prob().
		// Calculate final particle weight by multiplying all weights together.
		double weight = 1;
		for (int k = 0; k < observations_map.size(); ++k) {
			int lmrk_idx = observations_map[k].id - 1;
			//Protect for when particles go out of range
			if (lmrk_idx < 0) {
				weight = -1;
				break;
			}
			double obs_x = observations_map[k].x;
			double obs_y = observations_map[k].y;
			double lmrk_x = map_landmarks.landmark_list[lmrk_idx].x_f;
			double lmrk_y = map_landmarks.landmark_list[lmrk_idx].y_f;
			weight *= multiv_prob(std_landmark[0], std_landmark[1], obs_x, obs_y, lmrk_x, lmrk_y);
		}
		particles[i].weight = weight;
	}
	
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

	 // Creates random generator engine
	std::default_random_engine gen;
	// Distributions for integer and real numbers
	discrete_distribution<int> discr_dist(0, num_particles - 1);
	uniform_real_distribution<double> real_dist(0.0, 1.0);

	// generate random starting index for resampling wheel
	int index = discr_dist(gen);

	// Initialize beta to 0
	double beta = 0;

	// New particles vector
	vector<Particle> new_particles;

	// Assign all particle weights to a weights vector and find maximum
	double max_w = -1;
	for (int i = 0; i < num_particles; ++i) {
		weights[i] = particles[i].weight;
		if (particles[i].weight > max_w) {
			max_w = particles[i].weight;
		}
	}

	// Resampling wheel
	for (int i = 0; i < num_particles; ++i) {
		beta += real_dist(gen) * 2 * max_w;
		while (beta > weights[index]) {
			beta -= weights[index];
			if (index == num_particles - 1) {
				index = 0;
			}
			else {
				index = index + 1;
			}
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
	double mu_x, double mu_y) {
	// calculate normalization term
	double gauss_norm;
	gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

	// calculate exponent
	double exponent;
	exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
		+ (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

	// calculate weight using normalization terms and exponent
	double weight;
	weight = gauss_norm * exp(-exponent);

	return weight;
}