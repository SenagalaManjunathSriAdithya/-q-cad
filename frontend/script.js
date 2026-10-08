const analyzeButton = document.getElementById("analyzeButton");

const result = document.getElementById("result");

const classicalProbability =
    document.getElementById("classicalProbability");

const quantumProbability =
    document.getElementById("quantumProbability");

const hybridProbability =
    document.getElementById("hybridProbability");

const hybridRisk =
    document.getElementById("hybridRisk");

const hybridProgress =
    document.getElementById("hybridProgress");

const classicalProgress =
    document.getElementById("classicalProgress");

const quantumProgress =
    document.getElementById("quantumProgress");

const hybridProgressModel =
    document.getElementById("hybridProgressModel");

const thresholdValue =
    document.getElementById("thresholdValue");

const thresholdInfo =
    document.getElementById("thresholdInfo");


analyzeButton.addEventListener("click", async function () {

    const srcBytes =
        document.getElementById("src_bytes").value;

    const dstBytes =
        document.getElementById("dst_bytes").value;

    const count =
        document.getElementById("count").value;

    const dstHostSrvCount =
        document.getElementById("dst_host_srv_count").value;


    // Check inputs

    if (
        srcBytes === "" ||
        dstBytes === "" ||
        count === "" ||
        dstHostSrvCount === ""
    ) {

        result.innerHTML = `
            <div class="status anomaly">
                PLEASE ENTER ALL VALUES
            </div>
        `;

        return;
    }


    // Show analysis message

    result.innerHTML = `
        <div class="status waiting">
            ANALYZING TRAFFIC...
        </div>
    `;


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    src_bytes: Number(srcBytes),

                    dst_bytes: Number(dstBytes),

                    count: Number(count),

                    dst_host_srv_count:
                        Number(dstHostSrvCount)

                })
            }
        );


        const data = await response.json();


        if (!response.ok) {
            throw new Error(
                data.error || "Backend error"
            );
        }


        // Convert probabilities to percentages

        const classical =
            data.classical_probability * 100;

        const quantum =
            data.quantum_probability * 100;

        const hybrid =
            data.hybrid_probability * 100;

        const threshold =
            data.threshold * 100;


        // Display percentages

        classicalProbability.textContent =
            classical.toFixed(2) + "%";

        quantumProbability.textContent =
            quantum.toFixed(2) + "%";

        hybridProbability.textContent =
            hybrid.toFixed(2) + "%";


        // Hybrid risk

        hybridRisk.textContent =
            hybrid.toFixed(2) + "%";


        // Update progress bars

        classicalProgress.style.width =
            classical + "%";

        quantumProgress.style.width =
            quantum + "%";

        hybridProgressModel.style.width =
            hybrid + "%";

        hybridProgress.style.width =
            hybrid + "%";


        // Display threshold

        thresholdValue.textContent =
            threshold.toFixed(0) + "%";

        thresholdInfo.textContent =
            threshold.toFixed(0) + "%";


        // Final prediction

        if (data.prediction === "ANOMALY") {

            result.innerHTML = `
                <div class="status anomaly">
                    ⚠ ANOMALY DETECTED
                </div>

                <p>
                    The hybrid model classified this
                    network traffic as anomalous.
                </p>
            `;

        } else {

            result.innerHTML = `
                <div class="status normal">
                    ✓ NORMAL TRAFFIC
                </div>

                <p>
                    The hybrid model classified this
                    network traffic as normal.
                </p>
            `;

        }

    } catch (error) {

        console.error(error);

        result.innerHTML = `
            <div class="status anomaly">
                BACKEND CONNECTION ERROR
            </div>

            <p>
                Make sure the Q-CAD Flask server
                is running on port 5000.
            </p>
        `;

    }

});