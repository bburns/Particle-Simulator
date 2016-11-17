%-----------------------------------------------------------------------------------
% psim
% Particle Simulation Prototype
% This program simulates a number of particles confined to box of specified 
% dimensions, and calculates various thermodynamic properties.
%-----------------------------------------------------------------------------------
% Brian Burns
% Advanced Physics Lab
% Dr. John Miller, Jr. 
% April 2003
%-----------------------------------------------------------------------------------

clear; % clear all variables (helps catch variable misspellings, etc). 
clf % clear figure window 
clc % clear command window
close all; % delete any existing figures

% Append all output to log file
diary psimlog.txt;

ProgramVersion = 'psim, version April 2003';


%-----------------------------------------------------------------------------------
% Constants and Units
%-----------------------------------------------------------------------------------
% Mass:       Daltons (D) (1.66e-26 kg)
% Length:     Angstroms (A) (1e-10 m)
% Time:       Picosecs (ps) (1e-12 sec)
% Force:      Mewton "Molecular Newton" = DA/ps^2 
% Energy:     Moule "Molecular Joule" = DA^2/ps^2 
% Pressure:   Mascal "Molecular Pascal" = D/A/ps^2 
% Volume:     Molvo "Molecular Volume" = A^3 
%-----------------------------------------------------------------------------------

NA = 6.022e23; % Avogadro's number
kbJoules = 1.380658e-23; % Boltzmann's constant [Joules/K]
Dalton = 1.6605402e-27; % [kg]
Angstrom = 1e-10; % [meters]
ps = 1e-12; % [seconds]
kjpermol = 1000 / NA; % [1.66e-21 J]
eV = 1.602e-19; % [Joules]

Mewton = Dalton * Angstrom / ps^2; % [Newtons]
Moule = Dalton * Angstrom^2 / ps^2; % [Joules]
Mascal = Mewton / Angstrom^2; % [Pascals]
Molvo = Angstrom^3; % [m^3]

% Conversion Factors
NewtonsPerMewton = Mewton; % [1.66054e-13 Newtons/Mewton]
JoulesPerMoule = Moule; % [1.66054e-23 Joules/Moule]
PascalsPerMascal = Mascal; % [1.66054e7 Pascal/Mascal]
MewtonsPerNewton = 1/Mewton; % [6.022137377e12 Mewtons/Newton]
MoulesPerJoule = 1/Moule; % [6.022137e22 Moules/Joule]
MascalsPerPascal = 1/Mascal; % [6.022137e-8 Mascal/Pascal]
AngstromsPerMeter = 1/Angstrom;
KilogramsPerDalton = Dalton;
CubicMetersPerMolvo = Angstrom^3;
eVperMoule = Moule / eV; % [eV/Moule]
eVperJoule = 1/eV; % [eV/Joule]
AtmospheresPerPascal = 1/101325; % [Atm/Pa]
AtmospheresPerMascal = AtmospheresPerPascal * PascalsPerMascal; % [Atm/Mascal]

kb = kbJoules * MoulesPerJoule; % Boltzmann's constant [0.83145116 Moules/K]
g = 9.8 * AngstromsPerMeter / (1e12)^2; % Acceleration of gravity [9.8e-14 A/ps^2]
SpeedOfLight = 3e8 * 1e10/1e12; % [3e6 A/ps]


%-----------------------------------------------------------------------------------
% Initialization
%-----------------------------------------------------------------------------------
% Specify starting parameters here.
% A real program would read this info in from initialization files. 
% The different experiments are listed here.
%-----------------------------------------------------------------------------------

nd = 3; % number of dimensions
b3dDisplay = 1; % turn 3d display on / off

% Define different atomic species
Species(1).Name = 'Argon';
Species(1).Mass = 40; % [D]
Species(1).Radius = 0.95; % [A] 
Species(1).Color = 'b'; % blue
Species(1).Quantity = 0;
Species(2) = Species(1); % just a copy of species 1
Species(2).Color = 'g'; % green


% Maxwell Velocity Distribution
%--------------------------------------------------
% sExperiment='Maxwell Velocity Distribution';
% bGasCollisions = 1;
% InitTemperature = 100; % [K]
% sPotentialType = 'none'; % hard spheres / ideal gas
% nPotentialType = 0;
% sParticleDistribution = 'flat';
% sVelocityDistribution = 'same'; 
% DisplaySteps = 2; %round(1/TimeStep); % [steps/display]
% L = 10; % [A]    (range from 10-100 is good)
% bMembrane = 0;
% np = 200; % Number of particles - need enough for good histograms
% bVelocityHistogram = 1;
% TimeRange = 8; % [ps]
% TimeStep = 1/10; % [ps]
% nSpecies = 1;
% Species(1).Quantity = np;
% Species(1).Temperature = InitTemperature; 
% bShiftParticles = 0;
% SampleSteps = round(1/TimeStep); % number of time steps between samples. gives 1 sample per ps
% BoxSize = [L L L];
% nx=10;

% Ideal Gas Law
%--------------------------------------------------
% don't need gas collisions to verify ideal gas law - just use gaussian velocity distribution
% sExperiment='Ideal Gas Law';
% bGasCollisions = 0; 
% InitTemperature = 300; % [K]
% sPotentialType = 'none'; % hard spheres / ideal gas
% nPotentialType = 0;
% sParticleDistribution = 'flat';
% sVelocityDistribution = 'gaussian'; 
% DisplaySteps = 6; %round(1/TimeStep); % [steps/display]
% L = 20; % [A]    (range from 10-100 is good)
% BoxSize = [L L L];
% bMembrane = 0;
% np = 500; % Number of particles
% bVelocityHistogram = 1;
% TimeRange = 100; % [ps]
% TimeStep = 1/10; % [ps]
% nSpecies = 1;
% Species(1).Quantity = np;
% Species(1).Temperature = InitTemperature; 
% bShiftParticles = 0;
% SampleSteps = round(1/TimeStep); % number of time steps between samples. gives 1 sample per ps
% nx=10;

% Diffusion
%--------------------------------------------------
% sExperiment='Diffusion';
% bGasCollisions = 0; 
% InitTemperature = 100; % [K]
% sPotentialType = 'none'; % hard spheres / ideal gas
% nPotentialType = 0;
% sParticleDistribution = 'half';
% sVelocityDistribution = 'gaussian'; 
% DisplaySteps = 2; %round(1/TimeStep); % [steps/display]
% L = 20; % [A]
% BoxSize = [L L L];
% bMembrane = 0;
% np = 1000; % Number of particles
% bVelocityHistogram = 0;
% TimeRange = 10; % [ps]
% TimeStep = 1/40; % [ps]
% nSpecies = 2;
% Species(1).Quantity = np/2;
% Species(2).Quantity = np/2;
% bShiftParticles = 0;
% Species(1).Temperature = InitTemperature; 
% Species(2).Temperature = InitTemperature; 
% SampleSteps = round(1/TimeStep); % number of time steps between samples. gives 1 sample per ps
% nx=10;


% Diffusion2
%--------------------------------------------------
sExperiment='Diffusion';
bGasCollisions = 1; 
InitTemperature = 300; % [K]
sPotentialType = 'none'; % hard spheres / ideal gas
nPotentialType = 0;
sParticleDistribution = 'diff2';
sVelocityDistribution = 'gaussian2';
DisplaySteps = 2; %round(1/TimeStep); % [steps/display]
%L = 40; % [A]
BoxSize = [20 20 8]; % flat box
bMembrane = 0;
np = 220; % Number of particles
bVelocityHistogram = 0;
TimeRange = 10; % [ps]
TimeStep = 1/40; % [ps]
nSpecies = 2;
Species(1).Quantity = np*0.9;
Species(2).Quantity = np*0.1;
bShiftParticles = 0;
Species(1).Temperature = InitTemperature; 
Species(2).Temperature = InitTemperature; 
SampleSteps = round(1/TimeStep); % number of time steps between samples. gives 1 sample per ps
nx=10;

Species(2).Name = 'dye';
Species(2).Mass = 800; % [D]
Species(2).Radius = 4; % [A] 


% Heat Transfer
%--------------------------------------------------
% sExperiment='Heat Transfer';
% bGasCollisions = 1; 
% InitTemperature = 300; % [K]
% sPotentialType = 'none'; % hard spheres / ideal gas
% nPotentialType = 0;
% sParticleDistribution = 'half';
% sVelocityDistribution = 'gaussian'; 
% DisplaySteps = 4; %round(1/TimeStep); % [steps/display]
% BoxSize = [50 5 5]; % [A]
% bMembrane = 0;
% np = 300; % Number of particles
% bVelocityHistogram = 0;
% TimeRange = 10; % [ps]
% TimeStep = 1/25; % [ps]
% nSpecies = 2;
% Species(1).Quantity = np/2;
% Species(2).Quantity = np/2;
% Species(1).Temperature = InitTemperature; 
% Species(2).Temperature = 1; % cold part
% bShiftParticles = 0;
% SampleSteps = 5; %round(1/TimeStep); % number of time steps between samples. gives 1 sample per ps
% nx=20;

% Osmosis
%--------------------------------------------------
% semi-permeable membrane between halves of box (permeable to species 1 only)
% sExperiment='Osmosis';
% bGasCollisions = 0; 
% InitTemperature = 100; % [K]
% sPotentialType = 'none'; % hard spheres / ideal gas
% nPotentialType = 0;
% sParticleDistribution = 'half';
% sVelocityDistribution = 'gaussian'; 
% DisplaySteps = 2; %round(1/TimeStep); % [steps/display]
% L = 20; % [A]    (range from 10-100 is good)
% BoxSize = [L L L];
% bMembrane = 1;
% np = 500; % Number of particles
% bVelocityHistogram = 0;
% TimeRange = 10; % [ps]
% TimeStep = 1/40; % [ps]
% nSpecies = 2;
% Species(1).Quantity = np/2;
% Species(2).Quantity = np/2;
% bShiftParticles = 0;
% Species(1).Temperature = InitTemperature; 
% Species(2).Temperature = InitTemperature; 
% SampleSteps = round(1/TimeStep); % number of time steps between samples. gives 1 sample per ps
% nx=10;

% Real Gas Law
%--------------------------------------------------
% sExperiment='Real Gas Law';
% bGasCollisions = 0;
% sPotentialType = 'lennard-jones'; 
% nPotentialType = 1; % lennard-jones
% bVelocityHistogram = 0;
% InitTemperature = 400; % [K]
% sVelocityDistribution = 'gaussian';
% L = 20; % [A]
% BoxSize = [L L L];
% bMembrane = 0;
% %sParticleDistribution = 'test';
% %np = 2; % Number of particles
% npod = 3; % number of particles in one dimension 
% sParticleDistribution = 'fcc';
% %np = 20; % Number of particles
% np = npod^nd; % number of particles
% TimeRange = 200; % [ps]
% TimeStep = 0.01; % [ps]
% nSpecies = 1;
% Species(1).Quantity = np;
% bShiftParticles = 0;
% Species(1).Temperature = InitTemperature;
% SampleSteps = round(1/TimeStep); % number of time steps between samples. gives 1 sample per ps
% DisplaySteps = 5; 
% nx=10;



% Container
%-----------------------------------------------------------------------------------
BoxMatrix = [BoxSize(1) 0 0; 0 BoxSize(2) 0; 0 0 BoxSize(3)];
BoxArea = 2 * (BoxSize(1) * BoxSize(2) + BoxSize(2) * BoxSize(3) + ...
			BoxSize(1) * BoxSize(3)); % [A^2] ie 2LH + 2HW + 2LW
InitVolume = prod(BoxSize); % [A^3]



% Calculated Parameters
%-----------------------------------------------------------------------------------

TotalTimeSteps = TimeRange / TimeStep; % total number of time steps during run
SampleTime = TimeStep * SampleSteps; % amount of time elapsed between samples [ps]
TotalSamples = TotalTimeSteps / SampleSteps; % total number of samples to take during run

xrange = 0:(BoxSize(1)/nx):BoxSize(1);
trange = 0:SampleTime:TimeRange;
xrange = xrange(1:length(xrange)-1);
trange = trange(1:length(trange)-1);


% Assign data to arrays
%-----------------------------------------------------------------------------------

Radius = zeros(np,1);
Mass = zeros(np,1);
Color = zeros(np,1);
Position = zeros(np,nd);
Velocity = zeros(np,nd);
Acceleration = zeros(np,nd);

iStart = 1;
for i = 1:nSpecies
	
	AtomMass = Species(i).Mass;
	AtomRadius = Species(i).Radius;
	AtomVolume = 4/3*pi*AtomRadius^3; % [A^3]
	kT = kb * Species(i).Temperature; % Thermal energy [Moules]
	VxRms = sqrt(kT / AtomMass); % std deviation of one velocity component [A/ps]

	Quantity = Species(i).Quantity; % number of atoms of this species
	
	iEnd = iStart + Quantity - 1;
	Range = iStart:iEnd; % range of indices for this species (eg 20:39)
	
	Radius(Range) = Species(i).Radius;
	Mass(Range) = Species(i).Mass;
	Color(Range) = Species(i).Color;
	Species(i).Range = Range;
	r = Species(i).Radius;
	BoxMatrixInner = BoxMatrix - 2 * r * eye(3);
	
	% Particle Distribution
	% Distribute particles randomly through container (keep away from walls by radius of atom though)
	switch sParticleDistribution
	case 'half'
		% species 1 in left side of box
		% species 2 in right side of box
		% no barrier
		Position(Range,:) = rand(Quantity,nd) * BoxMatrixInner + r;
		Position(Range,1) = Position(Range,1) * 0.5 + BoxSize(1)*(i-1)/2; % shift to left or right side of the box
	case 'test'
		% two particles symmetrically located in box
		Position(1,:) = [BoxSize(1)/4, BoxSize(2)/2, BoxSize(3)/3];
		Position(2,:) = [3 * BoxSize(1)/4, BoxSize(2)/2, BoxSize(3)/3];
	case 'fcc'
		% arrange particles evenly throughout box volume
		for ix=1:npod
			for iy=1:npod
				for iz=1:npod
					n = iz + npod*(iy-1) + npod*npod*(ix-1);
					Position(n,1) = ix-1;
					Position(n,2) = iy-1;
					Position(n,3) = iz-1;
				end
			end
		end
		Position = Position / npod * BoxMatrix + BoxSize(1)/npod/2;
	case 'diff2'
		% simulate diffusion experiment:
		% species 1 - random
		% species 2 - circular distrib at center of box
		if i==1
			Position(Range,:) = rand(Quantity,nd) * BoxMatrixInner + r;
			Position(Range,3) = ones(Quantity,1) * BoxSize(3)/2; % z's in middle
		else
			theta = rand(Quantity,1) * 2 * pi;
			rpos = rand(Quantity,1) * 3;
			for j = 1:Quantity
				n = j+Range(1) - 1;
				Position(n,:) = [BoxSize(1)/2 + rpos(j)*cos(theta(j)), BoxSize(2)/2 + rpos(j)*sin(theta(j)), BoxSize(3)/2];
%				Position(Range,:) = randn(Quantity,nd) * 5 + BoxSize(1)/2	
			end
			Position(Range,3) = ones(Quantity,1) * BoxSize(3)/2; % z's in middle
		end
	otherwise
		Position(Range,:) = rand(Quantity,nd) * BoxMatrixInner + r;
	end
	
	% Velocity Distribution
	switch sVelocityDistribution
	case 'gaussian'
		% randn = normal (gaussian) distribution - multiply by std dev and add mean to get required distribution.
		Velocity(Range,:) = VxRms * randn(Quantity,nd); 
		% now need to make sure the net energy is actually InitTemp - need to scale v distribution to get it to match.
		% otherwise the random numbers may give you a temperature different from what you expected.
		for j=1:nd
			ActualRms = sqrt(mean(Velocity(Range,j) .* Velocity(Range,j))); % eg 1.96 should be 2.03
			Scale = VxRms / ActualRms;
			Velocity(Range,j) = Velocity(Range,j) * Scale;
		end
	case 'gaussian2'
		% same as 'gaussian' but set z velocity to zero
		% randn = normal (gaussian) distribution - multiply by std dev and add mean to get required distribution.
		Velocity(Range,:) = VxRms * randn(Quantity,nd); 
		% now need to make sure the net energy is actually InitTemp - need to scale v distribution to get it to match.
		% otherwise the random numbers may give you a temperature different from what you expected.
		for j=1:nd
			ActualRms = sqrt(mean(Velocity(Range,j) .* Velocity(Range,j))); % eg 1.96 should be 2.03
			Scale = VxRms / ActualRms;
			Velocity(Range,j) = Velocity(Range,j) * Scale;
		end
		Velocity(Range,3) = zeros(Quantity,1); % no z velocity
	case 'flat'
		Velocity(Range,:) = VxRms * (rand(Quantity,nd)-0.5); % flat distribution
	case 'same'
		% all have same speed but random directions
		Velocity(Range,:) = VxRms * sign(rand(Quantity,nd)-0.5); 
	case 'none'
		Velocity(Range,:) = zeros(Quantity,nd);
	end
	
	iStart = iEnd + 1;
	
end

		
% Expected Values
%-----------------------------------------------------------------------------------

AverageIntermolecularSpacing = (InitVolume/np)^(1/nd); % [A]
SpeedRms = sqrt(3 * kT / AtomMass); % [A/ps]
ExpectedPressure = np * kT / InitVolume * AtmospheresPerMascal; % [atm]
ExpectedCollisionRate = 4 * pi * sqrt(2) * AtomRadius^2 * np * SpeedRms / InitVolume * 1000; % [collisions/mlc/fs]
ExpectedMeanFreeTime = 1 / ExpectedCollisionRate; % [ps]
ExpectedMeanFreePath = SpeedRms * ExpectedMeanFreeTime; % [A]

% Maxwell-Boltzmann probability distribution
if bVelocityHistogram
	Speed = sqrt(kT / AtomMass); % [A/ps]
	s = Speed * (0:1/50:5); % Speed range
	ExpectedSpeedDistribution = np/2 * 4 * pi * (AtomMass/(2*pi*kT))^(3/2) * (s.^2) .* exp(-0.5 * AtomMass * s.^2 / kT); 
	s2 = Speed * (-2:1/50:2); % 1 dim speed range
	ExpectedSpeedDistribution1d = np/2 * sqrt (AtomMass/(2*pi*kT)) * exp(-0.5 * AtomMass * s2.^2 / kT);
end

ExpectedMostProbableSpeed = sqrt(2 * kT / AtomMass);
ExpectedAverageSpeed = sqrt(8 * kT / pi / AtomMass);
ExpectedRmsSpeed = sqrt(3 * kT / AtomMass);
VolumeFraction = AtomVolume * np / InitVolume;
MassDensity = np * AtomMass / InitVolume * KilogramsPerDalton / CubicMetersPerMolvo; % [kg/m^3]


%-----------------------------------------------------------------------------------
% Start
%-----------------------------------------------------------------------------------

fprintf('------------------------------------------------------------------------------------------\n');
fprintf('%s\n', ProgramVersion);
fprintf('Particle Simulator\n');
fprintf('Current time: %s\n', datestr(now, 0));
fprintf('------------------------------------------------------------------------------------------\n');
fprintf('\n');
fprintf('Experiment:                        %s\n', sExperiment);
fprintf('Number of Particles:               %d\n', np);
fprintf('Time Range:                        %.1f ps\n', TimeRange);
fprintf('Time Step:                         %.3f ps\n', TimeStep);
fprintf('InitTemperature:                   %d K\n', InitTemperature);
fprintf('Velocity Distribution:             %s \n', sVelocityDistribution);
fprintf('Potential Type:                    %s \n', sPotentialType);
fprintf('InitRmsSpeed:                      %.1f A/ps\n', SpeedRms);
fprintf('BoxSize:                           %d x %d x %d A\n', BoxSize);
fprintf('Volume:                            %d A^3\n', InitVolume);
fprintf('Avg Intermolecular Spacing:        %.2f A\n', AverageIntermolecularSpacing);
%. for single species...
fprintf('AtomMass:                          %.1f D\n', AtomMass);
fprintf('AtomRadius:                        %.1f A\n', AtomRadius);
fprintf('VolumeFraction:                    %.2f (particle volume / total volume)\n', VolumeFraction);
fprintf('MassDensity:                       %.2f kg/m^3\n', MassDensity);

fprintf('\n');


% Shift Particles
%-----------------------------------------------------------------------------------
% make sure particles aren't colliding with each other...
if bShiftParticles
	fprintf('Shifting particles...\n');
	bDoneShifting = 0;
	while bDoneShifting == 0
		nShifts = 0;
		bDoneShifting = 1;
		for i = 1:np-1
			for j = i+1:np
				DistanceVector = Position(i,:) - Position(j,:);
				Distance = norm(DistanceVector);
				if Distance < (Radius(i) + Radius(j))
					Position(i,:) = Position(i,:) + Radius(i);
					Position(j,:) = Position(j,:) - Radius(j);
					nShifts = nShifts + 1;
					bDoneShifting = 0;
				end
			end
		end
		fprintf('    shifted %d particles\n', nShifts);
	end
	fprintf('    done.\n');
end


% Draw 3d Scene
%-----------------------------------------------------------------------------------
if b3dDisplay

	% Define 3d shape
	% Define the 8 vertices of the unit cube, then define the 6 faces of the box, 
	% then scale the vertices by the box dimensions.
	UnitCubeVertices = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1]; 
	BoxFaces = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
	BoxVertices = UnitCubeVertices * BoxMatrix;
	
	% Draw particles in 3d
	for i = 1:nSpecies
		MarkerStyle = ['.', Species(i).Color]; % string concatenation (eg to '.r')
		Range = Species(i).Range;
		MarkerSize = Species(i).Radius * 10;
		hPlot(i) = plot3(Position(Range,1), Position(Range,2), Position(Range,3), MarkerStyle); % x,y,z
		set(hPlot(i), 'MarkerSize', MarkerSize);
		hold on;
	end
	
	% Set axis labels
	title('psim');
	xlabel('x-axis (A)');
	ylabel('y-axis (A)');
	zlabel('z-axis (A)');
	
	% Create 3d container object for display
	hContainer = patch('Vertices',BoxVertices, 'Faces',BoxFaces, 'FaceColor','blue');
	alpha(0.25); % make transpart!
	
	% Set axis lengths
	AxisMin = 0;
	axis([AxisMin BoxSize(1) AxisMin BoxSize(2) AxisMin BoxSize(3)]);
	axis equal;
	axis vis3d; % freezes aspect ratio properties to enable rotation of 3-D objects
	grid on;
	
	% Show 3d figure
	view(3);
	camproj perspective;
	rotate3d on; % turn on mouse-based 3-D rotation
	set(gca,'CameraViewAngleMode','manual'); % prevents graph from being resized as rotated
	drawnow;
	
end


% Save initial velocity for histogram
if bVelocityHistogram
	vxInit = Velocity(:, 1);
	vInit = sqrt(sum(Velocity .* Velocity, 2));
end

% Initialize arrays
DataTime = zeros(TotalSamples,1);
DataWallCollisions = zeros(TotalSamples,1);
DataGasCollisions = zeros(TotalSamples,1);
DataPressure = zeros(TotalSamples,1);
DataTemperature = zeros(TotalSamples,1);
DataTotalEnergy = zeros(TotalSamples,1);
DataConcentration = zeros(TotalSamples,nx);
DataTemperatureDist = zeros(TotalSamples,nx);

% Initialize timer values
TimeWalls = 0;
TimeGas = 0;
TimeProperties = 0;
TimeDisplay = 0;
TimeIntegrate = 0;

DeltaMomentum = 0;
WallCollisions = 0;
GasCollisions = 0;

SampleCounter = 0;
DisplayCounter = 0;
SampleIndex = 0;
TimeStepMin = TimeStep;

Force = zeros(np,nd);
AccelerationOld = zeros(np,nd);
Potential = zeros(np,1);

t = 0.0; % start time

%-----------------------------------------------------------------------------------
% Main Loop
%-----------------------------------------------------------------------------------

fprintf('\nRunning...\n');

for nStep = 1:TotalTimeSteps

	% Find max speed in any one dimension
	%. check for opposite also
	MaxSpeed = max(max(abs(Velocity)));
	TimeStepNew = AtomRadius / MaxSpeed;
	if TimeStepNew < TimeStepMin
		TimeStepMin = TimeStepNew;
	end
	
	% Calculate Force Fields
	% -----------------------------------------------------------------------------------
	switch nPotentialType
	case 1 
		% Lennard-Jones 6-12 Potential
		%	U = 4 * Epsilon * [ (Sigma/r)^12 - (Sigma/r)^6 ] (experimentally determined)
		%	F = - grad U = - 4 * Epsilon * (-12 * (Sigma^12)/(r^14) + 6 * (Sigma^6)/(r^8)) * rvector 
		%				 = - 4 * Epsilon * (-12 * (Sigma/r)^12 + 6 * (Sigma/r)^6 ) * rvector / r^2
		%	Argon: Epsilon = 1.65e-21 J		Sigma = 3.4 A
		%	Water: Epsilon = 1.08e-21 J		Sigma = 3.2 A
		Epsilon = 99.6; % [Moules]
		Sigma = 3.4; % [Angstroms]
		Force = zeros(np,nd);
		Potential = zeros(np,1);
		for i = 1:np-1
			for j = i+1:np
				DistanceVector = Position(i,:)-Position(j,:);
				Distance = norm(DistanceVector);
				U = 4 * Epsilon * ((Sigma/Distance)^12 - 2 * (Sigma/Distance)^6);
				Fij = - 4 * Epsilon * (-12 * (Sigma/Distance)^12 + 6 * (Sigma/Distance)^6 ) * DistanceVector / Distance / Distance;
				Force(i,:) = Force(i,:) + Fij;
				Force(j,:) = Force(j,:) - Fij;
				Potential(i) = Potential(i) + 0.5 * U;
				Potential(j) = Potential(j) + 0.5 * U;
			end
			Acceleration(i,:) = Force(i,:)/Mass(i);
		end
		Acceleration(np,:) = Force(np,:)/Mass(np);
	end
	
	
	% Calculate Properties
	%-----------------------------------------------------------------------------------
	% Calculate various properties based on the particle distribution and velocity. 
	% Note: SampleTime = amount of time elapsed since last sample was taken [ps]
	% PlanarPressure (atm) ie specify plane, get array of pressure points for graphical output
	% Pressure [atm] is averaged over all surfaces
	if SampleCounter == 0 % take samples only every so often
		tic;
		SampleCounter = SampleSteps; % reset sample counter
		
		Volume = prod(BoxSize); % [A^3]
		NumericDensity = np / Volume; % [Atoms/A^3]
		Pressure = DeltaMomentum / BoxArea / SampleTime * PascalsPerMascal * AtmospheresPerPascal; % [atm]
		SpeedSquared = sum(Velocity .* Velocity, 2); % v squared = (vx)^2 + (vy)^2 + (vz)^2  [array, summed across rows]
		KineticEnergy = 0;
		PotentialEnergy = sum(Potential); % [Moules]
		KineticEnergy = 0.5 * SpeedSquared' * Mass; % [Moules] % sum of 0.5 * Mass(i) * SpeedSquared(i)
		Temperature = 2.0 / 3.0 * KineticEnergy / kb / np; % [K] . dif for diatomic gas?
		TotalEnergy = KineticEnergy + PotentialEnergy; % [Moules]

		% get concentration distribution of species 1
		conc = histc(Position(Species(1).Range,1),xrange);

		% get temperature distribution of all species
		% bin is an array that tells you which bin each particle went into (0 to nx)
		ke = zeros(nx,1);
		count = zeros(nx,1);
		for i=1:np
			nBin = floor(Position(i,1)/BoxSize(1) * nx) + 1; % 1 to nx
			if nBin < 1 
				nBin = 1;
			end
			if nBin > nx
				nBin = nx;
			end
			ke(nBin) = ke(nBin) + 0.5 * Mass(i) * SpeedSquared(i);
			count(nBin) = count(nBin) + 1;
		end
		for i = 1:nx
			temp(i) = 2.0 / 3.0 * ke(i) / kb / count(i); % [K]
		end
		
		% Save data to arrays
		SampleIndex = SampleIndex + 1;
		DataTime(SampleIndex) = t;
		DataWallCollisions(SampleIndex) = WallCollisions;
		DataGasCollisions(SampleIndex) = GasCollisions;
		DataPressure(SampleIndex) = Pressure;
		DataTemperature(SampleIndex) = Temperature;
		DataKineticEnergy(SampleIndex) = KineticEnergy;
		DataPotentialEnergy(SampleIndex) = PotentialEnergy;
		DataConcentration(SampleIndex,:) = conc';
		DataTemperatureDist(SampleIndex,:) = temp;
		
		% Clear cumulative variables
		DeltaMomentum = 0;
		WallCollisions = 0;
		GasCollisions = 0;
		
		TimeProperties = TimeProperties + toc;
	end
	SampleCounter = SampleCounter - 1;
	
	
	% Display Data
	%-----------------------------------------------------------------------------------
	if DisplayCounter == 0
		tic;
		DisplayCounter = DisplaySteps;
		if b3dDisplay
			for i = 1:nSpecies
				Range = Species(i).Range;
				set(hPlot(i),'XData',Position(Range,1), 'YData',Position(Range,2), 'ZData',Position(Range,3)); % update 3d display
			end
			drawnow;
		end
		TimeDisplay = TimeDisplay + toc;
	end
	DisplayCounter = DisplayCounter - 1;
	
	
	
	% Integrate 
	%-----------------------------------------------------------------------------------
	% Verlet Algorithm
	% we always need the last two values of Position to produce the next one...
	% [Thijssen p475]
	% x(t+h) = 2x(t) - x(t-h) + h^2*a(x(t))
	tic;
	if nStep == 1 
		% for first step, don't have previous position yet
		PositionNew = Position + (TimeStep * Velocity) + (0.5 * TimeStep * TimeStep * Acceleration);
		Velocity = Velocity + TimeStep * Acceleration;
	else
		PositionNew = 2 * Position - PositionOld + TimeStep * TimeStep * Acceleration;
		Velocity = (PositionNew - PositionOld) / 2 / TimeStep;
	end

	% save position array
	PositionOld = Position;
	Position = PositionNew;
	
	TimeIntegrate = TimeIntegrate + toc;

	
	% Handle Wall Collisions
	%-----------------------------------------------------------------------------------
	% Check for collision with walls
	% Flip momentum component normal to wall
	% These are cumulative variables, cleared after each sample taken. 
	tic;
	for i = 1:np % particles
		for j = 1:nd % dimensions
			if Position(i,j) < Radius(i)
				Velocity(i,j) = -Velocity(i,j);
				Position(i,j) = 2 * Radius(i) - Position(i,j); % reflect about x=r axis
				DeltaMomentum = DeltaMomentum + abs(2*Mass(i)*Velocity(i,j)); % [DA/ps]
				WallCollisions = WallCollisions + 1;
			elseif Position(i,j) > (BoxSize(j) - Radius(i))
				Velocity(i,j) = -Velocity(i,j);
				Position(i,j) = 2 * (BoxSize(j) - Radius(i)) - Position(i,j); % reflect about x=L-r axis
				DeltaMomentum = DeltaMomentum + abs(2*Mass(i)*Velocity(i,j)); % [DA/ps]
				WallCollisions = WallCollisions + 1;
			end
		end
	end
	TimeWalls = TimeWalls + toc;

	% Check for collision with semi-permeable membrane
	if bMembrane
		for i = Species(2).Range
			if Position(i,1) < (MembranePosition + Radius(i))
				Velocity(i,1) = -Velocity(i,1);
				Position(i,j) = 2 * (MembranePosition + Radius(i)) - Position(i,j); % reflect about x=m+r axis
			end
		end
	end			
	
	% Handle Gas Collisions
	%-----------------------------------------------------------------------------------
	% Check for collision with other particles
	% This is the slowest part of the program. 
	% Swap momentum components parallel to vector connecting particles
	%. need smarter collision detector - ie trace each back to point where just touching to find correct distance vector
	tic;
	if bGasCollisions
		for i = 1:np-1
			for j = i+1:np
				DistanceVector = Position(i,:)-Position(j,:);
				Distance = norm(DistanceVector); 
				if Distance < (Radius(i) + Radius(j))
					VelocityVector = Velocity(i,:) - Velocity(j,:);
					DistanceHat = DistanceVector ./ Distance;
					VelocityComponent = dot(DistanceHat, VelocityVector) * DistanceHat;
					Velocity(i,:) = Velocity(i,:) - VelocityComponent;
					Velocity(j,:) = Velocity(j,:) + VelocityComponent;
					GasCollisions = GasCollisions + 1;
				end
			end
		end
	end
	TimeGas = TimeGas + toc;

	
	% Update current time
	t = t + TimeStep;
	
end




% Final calculations
%-----------------------------------------------------------------------------------
AveragePressure = mean(DataPressure); % [atm]
StdDevPressure = std(DataPressure);
AverageTemperature = mean(DataTemperature); % [K]
AverageCollisionRate = mean(DataGasCollisions) / np * 1000; % [collisions/mlc/fs]

% Save final velocity
if bVelocityHistogram
	vxFinal = Velocity(:, 1);
	vFinal = sqrt(sum(Velocity .* Velocity, 2));
end


% Display properties
%-----------------------------------------------------------------------------------

fprintf('\n\n');

fprintf('Results\n');
fprintf('                       Expected          Measured (avg)   StdDev            \n');
fprintf('Pressure:              %9.2f          %9.2f    %9.2f   atm               \n', ExpectedPressure, AveragePressure, StdDevPressure);
fprintf('Temperature:           %9.1f          %9.1f                K                 \n', InitTemperature, AverageTemperature);
fprintf('CollisionRate:         %9.2f          %9.2f                collisions/mlc/fs \n', ExpectedCollisionRate, AverageCollisionRate);
fprintf('\n\n');


fprintf('Time Requirements:\n');
fprintf('    Wall Collisions:      %.1f sec\n', TimeWalls);
fprintf('    Gas Collisions:       %.1f sec\n', TimeGas);
fprintf('    Calculate Properties: %.1f sec\n', TimeProperties);
fprintf('    Display Data:         %.1f sec\n', TimeDisplay);
fprintf('    Integrate:            %.1f sec\n', TimeIntegrate);
fprintf('\n\n');

if TimeStepMin < TimeStep
	fprintf('Need to reduce time step! currently %.3f ps, should be %.3f ps\n', TimeStep, TimeStepMin);
end



% Display figures
%-----------------------------------------------------------------------------------

% ScreenSize is a four-element vector: [left, bottom, width, height]:
Screensize = get(0,'ScreenSize');
Pos = [20 40 Screensize(3)*.8 Screensize(4)*.8];
figure('Position', Pos)
%figure;

warning off; % turn off warning messages

subplot 331;
plot(DataTime, DataWallCollisions);
xlabel('Time (ps)');
ylabel('Wall collisions');

subplot 332;
plot(DataTime, DataGasCollisions);
xlabel('Time (ps)');
ylabel('Gas collisions');

subplot 334;
plot(DataTime, DataPressure);
%hold on;
%plot(PredictedPressure);
xlabel('Time (ps)');
ylabel('Pressure (atm)');

subplot 335;
plot(DataTime, DataTemperature);
xlabel('Time (ps)');
ylabel('Temperature (K)');

subplot 336;
plot(DataTime, DataKineticEnergy,'g');
hold on;
plot(DataTime, DataPotentialEnergy,'r');
hold on;
plot(DataTime, DataKineticEnergy + DataPotentialEnergy,'b');
xlabel('Time (ps)');
ylabel('Energy (Moules)');
legend('ke','pe','net');

subplot 337;
surf(xrange,trange,DataConcentration);
xlabel('x (A)');
ylabel('time (ps)');
zlabel('conc');

subplot 338;
surf(xrange,trange,DataTemperatureDist);
xlabel('x (A)');
ylabel('time (ps)');
zlabel('temp (K)');

camproj perspective;
rotate3d on; % turn on mouse-based 3-D rotation


% Compare velocity distributions
%-----------------------------------------------------------------------------------
if bVelocityHistogram
	figure;
	x = 18;
	
	subplot 221;
	hist(vxInit, x);
	title('vx init');
	
	subplot 223;
	hist(vInit, x);
	title('v init');
	
	subplot 222;
	hist(vxFinal, x);
	title('vx final');
	% plot against maxwell-boltzmann velocity distribution
	hold on;
	plot(s2,ExpectedSpeedDistribution1d);
	
	subplot 224;
	hist(vFinal, x);
	title('v final');
	% plot against maxwell-boltzmann velocity distribution
	hold on;
	plot(s,ExpectedSpeedDistribution);
	
	% change color
	h = findobj(gcf,'Type','patch');
	set(h,'FaceColor','g','EdgeColor','w');
	
end


fprintf('\n');

% Turn off output log
diary off;

% Save all variables to file
%save psimvars.mat;
%fprintf('All data has been saved to psimvars.mat. Type \"load psimvars.mat\" and \"whos\" to examine data.\n');
%fprintf('Rename file if you want to keep data.\n');


%-----------------------------------------------------------------------------------
% Matlab Commands
%-----------------------------------------------------------------------------------
% An explanation of some Matlab commands: 
% view(3) - sets the default three-dimensional view, az = -37.5, el = 30.
% axis equal - sets the aspect ratio so that the data units are the same in every direction.
% axis square - makes the current axes region square (or cubed when three-dimensional). 
% rand(m,n) - returns an m-by-n matrix of random entries (0 to 1)
% randn - gives a normally distributed array of random numbers (mean 0, std dev 1)
% drawnow - flushes the event queue and updates the figure window.
% B = zeros(m,n) or B = zeros([m n]) - returns an m-by-n matrix of zeros.
% DistanceSquared = sum(DistanceVector .* DistanceVector, 2); % ie = dx^2 + dy^2 + dz^2
% norm(DistanceVector) = sqrt(sum(DistanceVector .* DistanceVector, 2))
%-----------------------------------------------------------------------------------

