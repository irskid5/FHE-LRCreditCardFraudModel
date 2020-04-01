#include "examples.h"

using namespace std;
using namespace seal;

class Model
{
public:
	int degree;
	double scale;
	shared_ptr<SEALContext> ctx;
	PublicKey publicKey;
	SecretKey secretKey;
	RelinKeys relinKeys;
	GaloisKeys galoisKeys;
	GaloisKeys galoisKeysFull;

	vector<vector<double>> weights1;
	vector<double> biases1;
	vector<double> weights2;
	vector<double> biases2;

	vector<double> coef;

	vector<vector<double>> test_features;
	vector<double> test_labels;

	vector<Ciphertext> x;

public:
	
	Model()
	{
		cout << "Default Constructor called" << endl;
		// Set up params
		EncryptionParameters parms(scheme_type::CKKS);
		degree = 15;
		int depth = ceil(log2(degree));
		vector<int> moduli(depth + 4 + 7, 60);
		moduli[0] = 40;
		moduli[moduli.size() - 1] = 40;
		size_t poly_modulus_degree = 16384*2;
		parms.set_poly_modulus_degree(poly_modulus_degree);
		parms.set_coeff_modulus(CoeffModulus::Create(
			poly_modulus_degree, moduli));
		
		// Set up scale
		scale = pow(2.0, 60);
		
		// Set up context
		ctx = SEALContext::Create(parms);

		print_parameters(ctx);
		cout << endl;

		cout << ctx->parameters_set() << endl;

		cout << "Generating keys..." << endl;
		KeyGenerator keygen(ctx);
		publicKey = keygen.public_key();
		secretKey = keygen.secret_key();
		relinKeys = keygen.relin_keys();
		vector<int> steps(29, 0);
		for (int i = 1; i < 30; i++) {
			steps[i-1] = i;
		}
		galoisKeys = keygen.galois_keys(steps);
		galoisKeysFull = keygen.galois_keys();

		cout << "DONE" << endl;
		
		cout << "Public Keys Size (Bytes) = " << sizeof(publicKey) << endl;
		cout << "Private Keys Size (Bytes) = " << sizeof(secretKey) << endl;
		cout << "Relin Keys Size (Bytes) = " << sizeof(relinKeys) << endl;
		cout << "Galois Keys Size (Bytes) = " << sizeof(galoisKeys) << endl;

		string filename = "D:\\Documents\\University Files\\Year 4\\Thesis\\dev\\KerasMLModel\\model1t0.04clip1\\realCoefficients\\weights1.csv";
		importMatrix(filename, ',' , weights1);

		filename = "D:\\Documents\\University Files\\Year 4\\Thesis\\dev\\KerasMLModel\\model1t0.04clip1\\realCoefficients\\biases1.csv";
		importVector(filename, ',', biases1);

		filename = "D:\\Documents\\University Files\\Year 4\\Thesis\\dev\\KerasMLModel\\model1t0.04clip1\\realCoefficients\\weights2.csv";
		importVector(filename, ',', weights2);

		filename = "D:\\Documents\\University Files\\Year 4\\Thesis\\dev\\KerasMLModel\\model1t0.04clip1\\realCoefficients\\biases2.csv";
		importVector(filename, ',', biases2);

		filename = "D:\\Documents\\University Files\\Year 4\\Thesis\\dev\\KerasMLModel\\model1t0.04clip1\\realCoefficients\\coef.csv";
		importVector(filename, ',', coef);

		filename = "D:\\Documents\\University Files\\Year 4\\Thesis\\dev\\KerasMLModel\\model1t0.04clip1\\realCoefficients\\test_features.csv";
		importMatrix(filename, ',', test_features);

		filename = "D:\\Documents\\University Files\\Year 4\\Thesis\\dev\\KerasMLModel\\model1t0.04clip1\\realCoefficients\\test_labels.csv";
		importVector(filename, ',', test_labels);

		cout << "DONE INIT" << endl;
	}

	void importMatrix(const std::string& filename, char sep, vector<vector<double>>& output)
	{
		std::ifstream src(filename);

		if (!src)
		{
			std::cerr << "\aError opening file.\n\n";
			exit(EXIT_FAILURE);
		}
		string buffer;
		while(getline(src, buffer))
		{
			size_t strpos = 0;
			size_t endpos = buffer.find(sep);
			vector<double> row;
			while (endpos < buffer.length())
			{  
				string numberStr = buffer.substr(strpos, endpos - strpos);
				double number = stod(numberStr);
				row.push_back(number);
				strpos = endpos + 1;
				endpos = buffer.find(sep, strpos);
			}
			string numberStr = buffer.substr(strpos);
			double number = stod(numberStr);
			row.push_back(number);
			output.push_back(row);
		}
	}

	void importVector(const std::string& filename, char sep, vector<double>& output)
	{
		std::ifstream src(filename);

		if (!src)
		{
			std::cerr << "\aError opening file.\n\n";
			exit(EXIT_FAILURE);
		}
		string buffer;
		while (getline(src, buffer))
		{
			string numberStr = buffer.substr(0);
			double number = stod(numberStr);
			output.push_back(number);
		}
	}

	void compute_all_powers(const Ciphertext &ctx, int degree, Evaluator &evaluator, RelinKeys &relin_keys, vector<Ciphertext> &powers) {

		powers.resize(degree + 1);
		powers[1] = ctx;

		vector<int> levels(degree + 1, 0);
		levels[1] = 0;
		levels[0] = 0;

		for (int i = 2; i <= degree; i++) {
			// compute x^i 
			int minlevel = i;
			int cand = -1;
			for (int j = 1; j <= i / 2; j++) {
				int k = i - j;
				//
				int newlevel = max(levels[j], levels[k]) + 1;
				if (newlevel < minlevel) {
					cand = j;
					minlevel = newlevel;
				}
			}
			levels[i] = minlevel;
			// use cand 
			if (cand < 0) throw runtime_error("error");
			//cout << "levels " << i << " = " << levels[i] << endl; 
			// cand <= i - cand by definition 
			Ciphertext temp = powers[cand];
			evaluator.mod_switch_to_inplace(temp, powers[i - cand].parms_id());

			evaluator.multiply(temp, powers[i - cand], powers[i]);
			evaluator.relinearize_inplace(powers[i], relin_keys);
			evaluator.rescale_to_next_inplace(powers[i]);
		}
		return;
	}

	void dotProductPlain(Plaintext &ptxt, Ciphertext &input, int dim, 
		Evaluator &evaluator, int slotsCount, Ciphertext &destination) {
		
		cout << "Beginning dot product evaluation..." << endl;
		
		auto t1 = std::chrono::high_resolution_clock::now();

		// Perform plain-cipher multiplication 
		evaluator.mod_switch_to_inplace(ptxt, input.parms_id());
		evaluator.multiply_plain(input, ptxt, destination);
		evaluator.rescale_to_next_inplace(destination);
		destination.scale() = scale;

		// Rotate and add destination vector to get dot product
		Ciphertext temp;
		for (size_t i = 1; i <= slotsCount/2; i<<=1) {
			evaluator.rotate_vector(destination, i, galoisKeysFull, temp);
			if (i == 0) {
				destination = temp;
			}
			else {
				evaluator.add_inplace(destination, temp);
			}
		}

		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time it took to run dotProductPlain (microSec): " + to_string(duration) << endl;
		
		return;
	}

	void matmul(vector<Plaintext> ptxt_diag, Ciphertext ct_v, Ciphertext &enc_result, int dim, Evaluator &evaluator) {

		cout << "Beginning matmul Operation..." << endl;
		
		auto t1 = std::chrono::high_resolution_clock::now();

		// Perform the multiplication 
		Ciphertext temp;
		for (int i = 0; i < dim; i++) {
			// rotate 
			evaluator.rotate_vector(ct_v, i, galoisKeys, temp);
			// multiply
			evaluator.multiply_plain_inplace(temp, ptxt_diag[i]);
			evaluator.rescale_to_next_inplace(temp);
			temp.scale() = scale;
			if (i == 0) {
				enc_result = temp;
			}
			else {
				evaluator.add_inplace(enc_result, temp);
			}
		}

		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		cout << "Time it took to run matmul (microSec): " + to_string(duration) << endl;

		return;
	}

	void encodeMatrixIntoDiag(vector<Plaintext> &ptxt_diag, CKKSEncoder &encoder, int dim, vector<vector<double>> M, double scale) {
		// Encode the diagonals
		for (int i = 0; i < dim; i++) {
			vector<double> diag(dim);
			for (int j = 0; j < dim; j++) {
				diag[j] = M[j][(j + i) % dim];
			}
			encoder.encode(diag, scale, ptxt_diag[i]);
		}
		return;
	}

	void polyeval_tree(int degree, Ciphertext &enc_result, Ciphertext &ctx, CKKSEncoder &encoder, Encryptor &encryptor, 
		vector<double> &coeffs, Evaluator &evaluator, Decryptor &decryptor) {

		cout << "Beginning polyeval Operation..." << endl;
		
		chrono::high_resolution_clock::time_point time_start, time_end;
		chrono::microseconds time_diff;

		vector<Plaintext> plain_coeffs(degree + 1);

		//cout << "Poly = ";
		for (size_t i = 0; i < degree + 1; i++) {
			// coeffs[i] = (double)rand() / RAND_MAX;
			encoder.encode(coeffs[i], scale, plain_coeffs[i]);
			/*vector<double> tmp;
			encoder.decode(plain_coeffs[i], tmp);
			cout << "Real = " << coeffs[i] << ", " << endl;
			cout << "Decoded = " << tmp[0] << ", " << endl;*/
		}
		//cout << endl;
		//cout << "encryption done " << endl;

		// compute all powers
		vector<Ciphertext> powers(degree + 1);

		time_start = chrono::high_resolution_clock::now();

		compute_all_powers(ctx, degree, evaluator, relinKeys, powers);
		//cout << "All powers computed " << endl;

		// result =a[0]
		encryptor.encrypt(plain_coeffs[0], enc_result);
		
		//for (int i = 1; i <= degree; i++){
		//	decryptor.decrypt(powers[i], plain_result);
		//	encoder.decode(plain_result, result);
		//	cout << "power  = " << result[0] << endl;
		//}
		
		// result += a[i]*x[i]
		for (int i = 1; i <= degree; i++) {
			// Even coeff (except 0 and 2) are zero, continue to next to avoid transparent ctx 
			if ((i % 2 == 0) && (i != 0) && (i != 2)) {
				continue;
			}
			// Continue with algo
			//cout << i << "-th sum started" << endl; 
			Ciphertext temp;
			evaluator.mod_switch_to_inplace(plain_coeffs[i], powers[i].parms_id());
			evaluator.multiply_plain(powers[i], plain_coeffs[i], temp);
			evaluator.rescale_to_next_inplace(temp);
			//cout << "got here " << endl; 
			evaluator.mod_switch_to_inplace(enc_result, temp.parms_id());
			enc_result.scale() = scale;
			temp.scale() = scale;
			evaluator.add_inplace(enc_result, temp);
			//cout << i << "-th sum done" << endl; 
		}
		time_end = chrono::high_resolution_clock::now();
		time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
		cout << "Done Polyeval [" << time_diff.count() << " microseconds]" << endl;
	}
};

void testModel() {
	Model mdl;

	CKKSEncoder encoder(mdl.ctx);
	Encryptor encryptor(mdl.ctx, mdl.publicKey);
	Decryptor decryptor(mdl.ctx, mdl.secretKey);
	Evaluator evaluator(mdl.ctx);

	int dim = 29;
	int degree = mdl.degree;

	Plaintext plain_result;
	vector<double> result;
	vector<double> predictions(mdl.test_labels.size(), -1);

	double loss = 0;

	// encode into diagonals
	vector<Plaintext> ptxt_diag(dim);
	mdl.encodeMatrixIntoDiag(ptxt_diag, encoder, dim, mdl.weights1, mdl.scale);

	for (int s = 0; s < 100; s++) {
		
		cout << "-------------------------- Starting s = " << s << " prediction --------------------------" << endl;
		
		vector<double> v = mdl.test_features[s];

		// Plaintext computation 
		//vector<double> resReal(dim, 0);
		//for (int i = 0; i < mdl.weights1.size(); i++) {
		//	for (int j = 0; j < dim; j++) {
		//		resReal[i] += mdl.weights1[i][j] * v[j];
		//	}
		//	resReal[i] += mdl.biases1[i];
		//}

		// repeat v throughout plaintext
		Plaintext ptxt_vec;
		vector<double> vrep(encoder.slot_count());
		for (int i = 0; i < vrep.size(); i++) vrep[i] = v[i % v.size()];
		encoder.encode(vrep, mdl.scale, ptxt_vec);

		// encrypt v
		Ciphertext ctv;
		encryptor.encrypt(ptxt_vec, ctv);

		// Perform matmul of ctv and diag plaintext vector
		Ciphertext enc_result;
		mdl.matmul(ptxt_diag, ctv, enc_result, dim, evaluator);

		// Add bias 1 to answer
		Plaintext biases1_plain;
		encoder.encode(mdl.biases1, mdl.scale, biases1_plain);
		evaluator.mod_switch_to_inplace(biases1_plain, enc_result.parms_id());
		evaluator.add_plain_inplace(enc_result, biases1_plain);

		// Verify result
		//decryptor.decrypt(enc_result, plain_result);
		//encoder.decode(plain_result, result);

		//for (int i = 0; i < dim; i++) {
		//	cout << "actual: " << result[i] << ", expected: " << resReal[i] << endl;
		//}

		// Go through first kernel
		Ciphertext result_ctxt;
		mdl.polyeval_tree(degree, result_ctxt, enc_result, encoder, encryptor, mdl.coef, evaluator, decryptor);

		// Verify result 
		//vector<double> expected_result(dim, 0);
		//for (int k = 0; k < dim; k++) {
		//	double expected_result_sing = mdl.coef[degree];
		//	for (int i = degree - 1; i >= 0; i--) {
		//		expected_result_sing *= result[k];
		//		expected_result_sing += mdl.coef[i];
		//	}
		//	expected_result[k] = expected_result_sing;
		//	cout << "Expected Result[" << k << "] = " << expected_result[k] << endl;
		//}

		//decryptor.decrypt(result_ctxt, plain_result);
		//encoder.decode(plain_result, result);

		//for (int i = 0; i < dim; i++) {
		//	cout << "Actual : " << result[i] << ", Expected : " << expected_result[i] << ", diff : " << abs(result[i] - expected_result[i]) << endl;
		//}

		// Go through second layer dot product

		/*double dotProduct = 0;
		for (int i = 0; i < dim; i++) {
			dotProduct += result[i] * mdl.weights2[i];
		}
		dotProduct += mdl.biases2[0];*/

		Plaintext weights2_plain;
		encoder.encode(mdl.weights2, mdl.scale, weights2_plain);
		Ciphertext destination;
		mdl.dotProductPlain(weights2_plain, result_ctxt, dim, evaluator, encoder.slot_count(), destination);

		// Add bias 2
		Plaintext biases2_plain;
		encoder.encode(mdl.biases2, mdl.scale, biases2_plain);
		evaluator.mod_switch_to_inplace(biases2_plain, destination.parms_id());
		evaluator.add_plain_inplace(destination, biases2_plain);

		// Verify result
		/*decryptor.decrypt(destination, plain_result);
		encoder.decode(plain_result, result);

		cout << "Actual : " << result[0] << ", Expected : " << dotProduct << ", diff : " << abs(result[0] - dotProduct) << endl;*/

		//double finalClass = mdl.coef[degree];
		//for (int i = degree - 1; i >= 0; i--) {
		//	finalClass *= result[0];
		//	finalClass += mdl.coef[i];
		//}

		// Go through last kernel
		Ciphertext finalResult;
		mdl.polyeval_tree(degree, finalResult, destination, encoder, encryptor, mdl.coef, evaluator, decryptor);
		
		// Verify result
		decryptor.decrypt(finalResult, plain_result);
		encoder.decode(plain_result, result);

		//cout << "Actual final class = " << result[0] << ", Expected final class = " << finalClass << endl;

		predictions[s] = result[0];

		loss += pow(predictions[s] - mdl.test_labels[s], 2);
		cout << "Difference = " << predictions[s] - mdl.test_labels[s] << endl;
		cout << "Loss = " << loss << endl;
	}
}