const elVideo = document.getElementById('video');

navigator.getMedia = (navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia);


const cargarCamera = () => {
    navigator.getMedia(
        {
            video: true,
            audio: false
        },
        stream => elVideo.srcObject = stream,
        console.error
    )
}

let labeledFaceDesciptor;
let confirmFace;
let count = 0;

Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
    //faceapi.nets.ageGenderNet.loadFromUri('/models'),
    //faceapi.nets.faceExpressionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    //faceapi.nets.faceLandmark68TinyNet.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    //faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
]).then(async () => {
    labeledFaceDesciptor = await loadLabeledImages();
}
).then(cargarCamera)

elVideo.addEventListener('play', async () => {
    //const labeledFaceDesciptor = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDesciptor, 0.6)

    const canvas = faceapi.createCanvasFromMedia(elVideo)
    document.body.append(canvas)

    const displaySize = {
        width: elVideo.width,
        height: elVideo.height
    }
    faceapi.matchDimensions(canvas, displaySize)

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(elVideo).withFaceLandmarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const results = resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor))

        results.forEach((result, i) => {
            if (result.label != "unknown") {
                if(confirmFace == result.label){
                    count += 1;
                } else {
                    confirmFace = result.label;
                    count = 0;
                }

                if(count > 10){
                    console.log({
                        result: result.label,
                        count
                    });
                }
                const box = resizedDetections[i].detection.box
                const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
                drawBox.draw(canvas)
            } else {
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
            }
        });
    })
})

async function loadLabeledImages() {
    console.log(new Date());
    const labels = ['Santiago Galvez', 'Candelaria de Goycoechea', 'Chris Hemsworth', 'Robert Downey jr','Cris Evans']
    return Promise.all(
        labels.map(async label => {
            const descriptions = []
            for (let i = 1; i <= 2; i++) {
                //const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`)
                //const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/SantiagoGalvez00/Reconocimiento-facial/master/images/${label}/${i}.png`)
                const img = await faceapi.fetchImage(`images/${label}/${i}.png`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }
            console.log(new Date());

            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}