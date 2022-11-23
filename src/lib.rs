#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::upper_case_acronyms)]

pub mod cost;
pub mod loader;
pub mod pcs;
pub mod system;
pub mod util;
pub mod verifier;

#[derive(Clone, Debug)]
pub enum Error {
    InvalidInstances,
    InvalidLinearization,
    InvalidQuery(util::protocol::Query),
    InvalidChallenge(usize),
    AssertionFailure(String),
    Transcript(std::io::ErrorKind, String),
}

#[derive(Clone, Debug)]
pub struct Protocol<C, L = loader::native::NativeLoader>
where
    C: util::arithmetic::CurveAffine,
    L: loader::Loader<C>,
{
    // Common description
    pub domain: util::arithmetic::Domain<C::Scalar>,
    pub preprocessed: Vec<L::LoadedEcPoint>,
    pub num_instance: Vec<usize>,
    pub num_witness: Vec<usize>,
    pub num_challenge: Vec<usize>,
    pub evaluations: Vec<util::protocol::Query>,
    pub queries: Vec<util::protocol::Query>,
    pub quotient: util::protocol::QuotientPolynomial<C::Scalar>,
    // Minor customization
    pub transcript_initial_state: Option<L::LoadedScalar>,
    pub instance_committing_key: Option<util::protocol::InstanceCommittingKey<C>>,
    pub linearization: Option<util::protocol::LinearizationStrategy>,
    pub accumulator_indices: Vec<Vec<(usize, usize)>>,
}

#[cfg(test)]
mod test {
    use crate::system::halo2::{compile, Config};
    use halo2_curves::{
        bn256::{Bn256, Fr},
        pairing::{Engine, MultiMillerLoop},
    };
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{AssignedCell, Layouter, Region, SimpleFloorPlanner, Value},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
            ConstraintSystem, Error, ProvingKey, Selector, VerifyingKey,
        },
        poly::{
            commitment::{CommitmentScheme, Params, ParamsProver},
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::AccumulatorStrategy,
            },
            Rotation, VerificationStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    };
    use rand::Rng;
    use std::{fmt::Debug, fs::write, io::Read, time::Instant};

    /// Convert bytes array to a wide representation of 64 bytes
    pub fn to_wide(b: &[u8]) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        bytes[..b.len()].copy_from_slice(b);
        bytes
    }

    /// Generate parameters with polynomial degere = `k`.
    pub fn generate_params<E: MultiMillerLoop + Debug>(k: u32) -> ParamsKZG<E> {
        ParamsKZG::<E>::new(k)
    }

    /// Write parameters to a file.
    pub fn write_params<E: MultiMillerLoop + Debug>(params: &ParamsKZG<E>, path: &str) {
        let mut buffer: Vec<u8> = Vec::new();
        params.write(&mut buffer).unwrap();
        write(path, buffer).unwrap();
    }

    /// Read parameters from a file.
    pub fn read_params<E: MultiMillerLoop + Debug>(path: &str) -> ParamsKZG<E> {
        let mut buffer: Vec<u8> = Vec::new();
        let mut file = std::fs::File::open(path).unwrap();
        file.read_to_end(&mut buffer).unwrap();
        ParamsKZG::<E>::read(&mut &buffer[..]).unwrap()
    }

    /// Proving/verifying key generation.
    pub fn keygen<E: MultiMillerLoop + Debug, C: Circuit<E::Scalar>>(
        params: &ParamsKZG<E>,
        circuit: &C,
    ) -> Result<ProvingKey<<E as Engine>::G1Affine>, Error> {
        let vk = keygen_vk::<<E as Engine>::G1Affine, ParamsKZG<E>, _>(params, circuit)?;
        let pk = keygen_pk::<<E as Engine>::G1Affine, ParamsKZG<E>, _>(params, vk, circuit)?;

        Ok(pk)
    }

    /// Helper function for finalizing verification
    // Rust compiler can't infer the type, so we need to make a helper function
    pub fn finalize_verify<
        'a,
        E: MultiMillerLoop + Debug,
        V: VerificationStrategy<'a, KZGCommitmentScheme<E>, VerifierSHPLONK<'a, E>>,
    >(
        v: V,
    ) -> bool {
        v.finalize()
    }

    /// Make a proof for generic circuit.
    pub fn prove<E: MultiMillerLoop + Debug, C: Circuit<E::Scalar>, R: Rng + Clone>(
        params: &ParamsKZG<E>,
        circuit: C,
        pub_inps: &[&[<KZGCommitmentScheme<E> as CommitmentScheme>::Scalar]],
        pk: &ProvingKey<E::G1Affine>,
        rng: &mut R,
    ) -> Result<Vec<u8>, Error> {
        let mut transcript = Blake2bWrite::<_, E::G1Affine, Challenge255<_>>::init(vec![]);
        create_proof::<KZGCommitmentScheme<E>, ProverSHPLONK<_>, _, _, _, _>(
            params,
            pk,
            &[circuit],
            &[pub_inps],
            rng.clone(),
            &mut transcript,
        )?;

        let proof = transcript.finalize();
        Ok(proof)
    }

    /// Verify a proof for generic circuit.
    pub fn verify<E: MultiMillerLoop + Debug>(
        params: &ParamsKZG<E>,
        pub_inps: &[&[<KZGCommitmentScheme<E> as CommitmentScheme>::Scalar]],
        proof: &[u8],
        vk: &VerifyingKey<E::G1Affine>,
    ) -> Result<bool, Error> {
        let strategy = AccumulatorStrategy::<E>::new(params);
        let mut transcript = Blake2bRead::<_, E::G1Affine, Challenge255<_>>::init(proof);
        let output = verify_proof::<KZGCommitmentScheme<E>, VerifierSHPLONK<E>, _, _, _>(
            params,
            vk,
            strategy,
            &[pub_inps],
            &mut transcript,
        )?;

        Ok(finalize_verify(output))
    }

    /// Helper function for doing proof and verification at the same time.
    pub fn prove_and_verify<E: MultiMillerLoop + Debug, C: Circuit<E::Scalar>, R: Rng + Clone>(
        params: ParamsKZG<E>,
        circuit: C,
        pub_inps: &[&[<KZGCommitmentScheme<E> as CommitmentScheme>::Scalar]],
        rng: &mut R,
    ) -> Result<bool, Error> {
        let pk = keygen(&params, &circuit)?;
        let start = Instant::now();
        let proof = prove(&params, circuit, pub_inps, &pk, rng)?;
        let end = start.elapsed();
        print!("Proving time: {:?}", end);
        let res = verify(&params, pub_inps, &proof[..], pk.get_vk())?;

        Ok(res)
    }

    #[derive(Clone, Debug)]
    /// Configuration elements for the circuit defined here.
    pub struct AggConfig {
        /// Configures a column for the x.
        x: Column<Advice>,
        /// Configures a column for the y.
        y: Column<Advice>,
        /// Configures a fixed boolean value for each row of the circuit.
        selector: Selector,
    }

    /// Constructs individual cells for the configuration elements.
    pub struct AggChip<F: FieldExt> {
        /// Assigns a cell for x.
        x: AssignedCell<F, F>,
        /// Assigns a cell for y.
        y: AssignedCell<F, F>,
    }

    impl<F: FieldExt> AggChip<F> {
        /// Create a new chip.
        pub fn new(x: AssignedCell<F, F>, y: AssignedCell<F, F>) -> Self {
            AggChip { x, y }
        }

        /// Make the circuit config.
        pub fn configure(meta: &mut ConstraintSystem<F>) -> AggConfig {
            let x = meta.advice_column();
            let y = meta.advice_column();
            let s = meta.selector();

            meta.enable_equality(x);
            meta.enable_equality(y);

            meta.create_gate("mul", |v_cells| {
                let x_exp = v_cells.query_advice(x, Rotation::cur());
                let y_exp = v_cells.query_advice(y, Rotation::cur());
                let x_next_exp = v_cells.query_advice(x, Rotation::next());
                let s_exp = v_cells.query_selector(s);

                vec![
                    // (x * y) - z == 0
                    // z is the next rotation cell for the x value.
                    // Example:
                    // let x = 3;
                    // let y = 2;
                    // let z = (x * y);
                    // z;
                    //
                    // z = (3 * 2) = 6 => We check the constraint (3 * 2) - 6 == 0
                    s_exp * ((x_exp * y_exp) - x_next_exp),
                ]
            });

            AggConfig { x, y, selector: s }
        }

        /// Synthesize the circuit.
        pub fn synthesize(
            &self,
            config: AggConfig,
            mut layouter: impl Layouter<F>,
        ) -> Result<AssignedCell<F, F>, Error> {
            layouter.assign_region(
                || "Agg",
                |mut region: Region<'_, F>| {
                    config.selector.enable(&mut region, 0)?;
                    let assigned_x = self.x.copy_advice(|| "x", &mut region, config.x, 0)?;
                    let assigned_y = self.y.copy_advice(|| "y", &mut region, config.y, 0)?;

                    let out = assigned_x.value().cloned() * assigned_y.value();

                    let out_assigned = region.assign_advice(|| "out", config.x, 1, || out)?;

                    Ok(out_assigned)
                },
            )
        }
    }

    #[derive(Clone)]
    struct TestConfig {
        agg: AggConfig,
        temp: Column<Advice>,
    }

    #[derive(Clone)]
    struct TestCircuit<F: FieldExt> {
        x: F,
        y: F,
    }

    impl<F: FieldExt> TestCircuit<F> {
        fn new(x: F, y: F) -> Self {
            Self { x, y }
        }
    }

    impl<F: FieldExt> Circuit<F> for TestCircuit<F> {
        type Config = TestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> TestConfig {
            let agg = AggChip::configure(meta);
            let temp = meta.advice_column();

            meta.enable_equality(temp);

            TestConfig { agg, temp }
        }

        fn synthesize(
            &self,
            config: TestConfig,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let (x, y) = layouter.assign_region(
                || "temp",
                |mut region: Region<'_, F>| {
                    let x = region.assign_advice(
                        || "temp_x",
                        config.temp,
                        0,
                        || Value::known(self.x),
                    )?;
                    let y = region.assign_advice(
                        || "temp_y",
                        config.temp,
                        1,
                        || Value::known(self.y),
                    )?;
                    Ok((x, y))
                },
            )?;
            let agg_chip = AggChip::new(x, y);
            let res = agg_chip.synthesize(config.agg, layouter.namespace(|| "agg"))?;
            Ok(())
        }
    }

    #[test]
    fn should_call_compile() {
        // Testing x = 5, y = 2.
        let k = 4;
        let test_chip = TestCircuit::new(Fr::from(5), Fr::from(2));
        let setup = generate_params::<Bn256>(k);
        let pk = keygen(&setup, &test_chip);
        let accumulator_indices: Option<Vec<(usize, usize)>> =
            Some(vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
        let num_instance = vec![1];

        let config: Config = Config {
            zk: true,
            query_instance: true,
            num_proof: 2,
            num_instance,
            accumulator_indices,
        };

        let protocol = compile(&setup, pk.unwrap().get_vk(), config);

        println!("{:#?}", protocol.preprocessed);
        println!("{:#?}", protocol.instance_committing_key);
        println!("Evaluations = {:?}", protocol.evaluations);
        println!("Queries = {:?}", protocol.queries);
        println!("Quotient = {:?}", protocol.quotient);
    }
}
