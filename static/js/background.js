let scene, camera, renderer, particles;
let mouseX = 0;
let mouseY = 0;

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 30;

    renderer = new THREE.WebGLRenderer({ 
        canvas: document.getElementById('bg-canvas'),
        alpha: true 
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);

    // Create particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 1800;
    const posArray = new Float32Array(particlesCount * 3);
    const colors = new Float32Array(particlesCount * 3);
    const sizes = new Float32Array(particlesCount);

    for(let i = 0; i < particlesCount * 3; i += 3) {
        // Position
        posArray[i] = (Math.random() - 0.5) * 50;
        posArray[i + 1] = (Math.random() - 0.5) * 50;
        posArray[i + 2] = (Math.random() - 0.5) * 50;

        // Color - mix between orange and gray shades
        const mixFactor = Math.random();
        const colorChoice = Math.random();

        if (colorChoice < 0.4) {
            // Orange shade (255, 87, 34) to (255, 112, 67)
            colors[i] = 1.0;                        // R (255)
            colors[i + 1] = 0.341 + mixFactor * 0.098; // G (87-112)
            colors[i + 2] = 0.133 + mixFactor * 0.129; // B (34-67)
        } else if (colorChoice < 0.7) {
            // Light orange (255, 171, 145) to (255, 204, 188)
            colors[i] = 1.0;                        // R (255)
            colors[i + 1] = 0.671 + mixFactor * 0.129; // G (171-204)
            colors[i + 2] = 0.569 + mixFactor * 0.167; // B (145-188)
        } else {
            // Gray (66, 66, 66) to (96, 96, 96)
            const grayValue = 0.259 + mixFactor * 0.118; // (66-96)/255
            colors[i] = grayValue;     // R
            colors[i + 1] = grayValue; // G
            colors[i + 2] = grayValue; // B
        }

        // Size with variation
        sizes[i/3] = Math.random() * 2 + 0.5;
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particlesGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const particlesMaterial = new THREE.PointsMaterial({
        size: 0.2,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
    });

    particles = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particles);

    document.addEventListener('mousemove', onMouseMove);
    window.addEventListener('resize', onWindowResize);
}

function onMouseMove(event) {
    mouseX = (event.clientX - window.innerWidth / 2) * 0.0003;
    mouseY = (event.clientY - window.innerHeight / 2) * 0.0003;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);

    // Smooth rotation
    particles.rotation.x += (mouseY * 0.2 - particles.rotation.x) * 0.05;
    particles.rotation.y += (mouseX * 0.2 - particles.rotation.y) * 0.05;

    // Gentle constant rotation
    particles.rotation.x += 0.0005;
    particles.rotation.y += 0.0003;

    renderer.render(scene, camera);
}

document.addEventListener('DOMContentLoaded', () => {
    init();
    animate();
});
