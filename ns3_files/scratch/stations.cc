#include <chrono>
#include <filesystem>
#include <map>
#include <string>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/node-container.h"
#include "ns3/node-list.h"
#include "ns3/ssid.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("stations");

/***** Functions declarations *****/

void ChangePower (Ptr<Node> staNode, uint8_t powerLevel);
void GetWarmupFlows (Ptr<FlowMonitor> monitor);
void InstallTrafficGenerator (Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, uint32_t port,
                              DataRate offeredLoad, uint32_t packetSize, double warmupTime,
                              double simulationTime, double fuzzTime);
void PopulateARPcache ();
void PowerCallback (std::string path, Ptr<const Packet> packet, double txPowerW);
void UpdateDistance (Ptr<Node> staNode, Ptr<Node> apNode);

/***** Global variables and constants *****/

#define DISTANCE_UPDATE_INTERVAL 0.005

std::map<uint32_t, uint64_t> warmupFlows;
u_int8_t globalPowerLevel = 0;

/***** Main with scenario definition *****/

int
main (int argc, char *argv[])
{
  // Initialize default simulation parameters
  std::string rateAdaptationManager = "ns3::MlWifiManager";
  std::string wifiManagerName = "";

  uint32_t nWifi = 1;
  double distance = 0.;
  double fuzzTime = 5.;
  double warmupTime = 10.;
  double simulationTime = 50.;
  double delta = 15;      // difference between 2 power levels in dB
  double interval = 2.;   // mean (exponential) interval between power change

  std::string pcapName = "";
  std::string csvPath = "results.csv";

  bool ampdu = true;
  uint32_t packetSize = 1500;
  uint32_t dataRate = 125;
  uint32_t channelWidth = 20;
  uint32_t minGI = 3200;

  std::string lossModel = "Nakagami";
  std::string mobilityModel = "Distance";
  double area = 40.;
  double nodeSpeed = 1.4;
  double nodePause = 20.;

  // Parse command line arguments
  CommandLine cmd;
  cmd.AddValue ("ampdu", "Use AMPDU (boolean flag)", ampdu);
  cmd.AddValue ("area","Size of the square in which stations are wandering (m) - only for RWPM mobility type", area);
  cmd.AddValue ("channelWidth", "Channel width (MHz)", channelWidth);
  cmd.AddValue ("csvPath", "Path to output CSV file", csvPath);
  cmd.AddValue ("dataRate", "Traffic generator data rate (Mb/s)", dataRate);
  cmd.AddValue ("delta", "Power change (dBm)", delta);
  cmd.AddValue ("distance", "Distance between AP and STAs (m) - only for Distance mobility type",distance);
  cmd.AddValue ("fuzzTime", "Maximum fuzz value (s)", fuzzTime);
  cmd.AddValue ("interval", "Interval between power change (s)", interval);
  cmd.AddValue ("lossModel", "Propagation loss model (LogDistance, Nakagami)", lossModel);
  cmd.AddValue ("manager", "Rate adaptation manager", rateAdaptationManager);
  cmd.AddValue ("managerName", "Name of the Wi-Fi manager in CSV", wifiManagerName);
  cmd.AddValue ("minGI", "Shortest guard interval (ns)", minGI);
  cmd.AddValue ("mobilityModel", "Mobility model (Distance, RWPM)", mobilityModel);
  cmd.AddValue ("nodeSpeed", "Maximum station speed (m/s) - only for RWPM mobility type",nodeSpeed);
  cmd.AddValue ("nodePause","Maximum time station waits in newly selected position (s) - only for RWPM mobility type",nodePause);
  cmd.AddValue ("nWifi", "Number of stations", nWifi);
  cmd.AddValue ("packetSize", "Packets size (B)", packetSize);
  cmd.AddValue ("pcapName", "Name of a PCAP file generated from the AP", pcapName);
  cmd.AddValue ("simulationTime", "Duration of simulation (s)", simulationTime);
  cmd.AddValue ("warmupTime", "Duration of warmup stage (s)", warmupTime);
  cmd.Parse (argc, argv);

  if (wifiManagerName.empty ())
    {
      wifiManagerName = rateAdaptationManager;
    }

  // Print simulation settings to screen
  std::cout << std::endl
            << "Simulating an IEEE 802.11ax devices with the following settings:" << std::endl
            << "- frequency band: 5 GHz" << std::endl
            << "- max data rate: " << dataRate << " Mb/s" << std::endl
            << "- delta: " << delta << " dBm" << std::endl
            << "- interval: " << interval << " s" << std::endl
            << "- channel width: " << channelWidth << " Mhz" << std::endl
            << "- shortest guard interval: " << minGI << " ns" << std::endl
            << "- packets size: " << packetSize << " B" << std::endl
            << "- AMPDU: " << ampdu << std::endl
            << "- rate adaptation manager: " << rateAdaptationManager << std::endl
            << "- number of stations: " << nWifi << std::endl
            << "- simulation time: " << simulationTime << " s" << std::endl
            << "- warmup time: " << warmupTime << " s" << std::endl
            << "- max fuzz time: " << fuzzTime << " s" << std::endl
            << "- loss model: " << lossModel << std::endl;

  if (mobilityModel == "Distance")
    {
      std::cout << "- mobility model: " << mobilityModel << std::endl
                << "- distance: " << distance << " m" << std::endl
                << std::endl;
    }
  else if (mobilityModel == "RWPM")
    {
      std::cout << "- mobility model: " << mobilityModel << std::endl
                << "- area: " << area << " m" << std::endl
                << "- max node speed: " << nodeSpeed << " m/s" << std::endl
                << "- max node pause: " << nodePause << " s" << std::endl
                << std::endl;
    }

  // Create AP and stations
  NodeContainer wifiApNode (1);
  NodeContainer wifiStaNodes (nWifi);

  // Configure mobility
  MobilityHelper mobility;

  if (mobilityModel == "Distance")
    {
      mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
      mobility.Install (wifiApNode);
      mobility.Install (wifiStaNodes);

      // Place AP at (distance, 0)
      Ptr<MobilityModel> mobilityAp = wifiApNode.Get (0)->GetObject<MobilityModel> ();
      mobilityAp->SetPosition (Vector3D (distance, 0., 0.));
    }
  else if (mobilityModel == "RWPM")
    {
      // Place AP at (0, 0)
      mobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");
      mobility.Install (wifiApNode);

      // Place nodes randomly in square extending from (0, 0) to (area, area)
      ObjectFactory pos;
      pos.SetTypeId ("ns3::RandomRectanglePositionAllocator");
      std::stringstream ssArea;
      ssArea << "ns3::UniformRandomVariable[Min=0.0|Max=" << area << "|Stream=42]";
      pos.Set ("X", StringValue (ssArea.str ()));
      pos.Set ("Y", StringValue (ssArea.str ()));

      Ptr<PositionAllocator> taPositionAlloc = pos.Create ()->GetObject<PositionAllocator> ();
      mobility.SetPositionAllocator (taPositionAlloc);

      // Set random pause (from 0 to nodePause [s]) and speed (from 0 to nodeSpeed [m/s])
      std::stringstream ssSpeed;
      ssSpeed << "ns3::UniformRandomVariable[Min=0.0|Max=" << nodeSpeed << "|Stream=42]";
      std::stringstream ssPause;
      ssPause << "ns3::UniformRandomVariable[Min=0.0|Max=" << nodePause << "|Stream=42]";

      mobility.SetMobilityModel ("ns3::RandomWaypointMobilityModel",
                                 "Speed", StringValue (ssSpeed.str ()),
                                 "Pause", StringValue (ssPause.str ()),
                                 "PositionAllocator", PointerValue (taPositionAlloc));

      mobility.Install (wifiStaNodes);
    }
  else
    {
      std::cerr << "Selected incorrect mobility model!";
      return 2;
    }

  // Print position of each node
  std::cout << "Node positions:" << std::endl;

  // AP position
  Ptr<MobilityModel> position = wifiApNode.Get (0)->GetObject<MobilityModel> ();
  Vector pos = position->GetPosition ();
  std::cout << "AP:\tx=" << pos.x << ", y=" << pos.y << std::endl;

  // Stations positions
  for (auto node = wifiStaNodes.Begin (); node != wifiStaNodes.End (); ++node)
    {
      position = (*node)->GetObject<MobilityModel> ();
      pos = position->GetPosition ();
      std::cout << "Sta " << (*node)->GetId () << ":\tx=" << pos.x << ", y=" << pos.y << std::endl;
    }

  std::cout << std::endl;

  // Configure wireless channel
  YansWifiPhyHelper phy;
  YansWifiChannelHelper channelHelper = YansWifiChannelHelper::Default ();

  if (lossModel == "Nakagami")
    {
      // Add Nakagami fading to the default log distance model
      channelHelper.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
    }
  else if (lossModel != "LogDistance")
    {
      std::cerr << "Selected incorrect loss model!";
      return 1;
    }
  phy.SetChannel (channelHelper.Create ());

  // Configure two power levels
  phy.Set ("TxPowerLevels", UintegerValue (2));
  phy.Set ("TxPowerStart", DoubleValue (21.0 - delta));
  phy.Set ("TxPowerEnd", DoubleValue (21.0));

  // Configure MAC layer
  WifiMacHelper mac;
  WifiHelper wifi;

  wifi.SetStandard (WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager (rateAdaptationManager);

  //Set SSID
  Ssid ssid = Ssid ("ns3-80211ax");
  mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "MaxMissedBeacons",
               UintegerValue (1000)); // prevents exhaustion of association IDs

  // Create and configure Wi-Fi interfaces
  NetDeviceContainer staDevice;
  staDevice = wifi.Install (phy, mac, wifiStaNodes);

  mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  // Manage AMPDU aggregation
  if (!ampdu)
    {
      Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_MaxAmpduSize",
                   UintegerValue (0));
    }

  // Set channel width and shortest GI
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelSettings",
               StringValue ("{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}"));

  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
               TimeValue (NanoSeconds (minGI)));

  // Install an Internet stack
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNodes);

  // Configure IP addressing
  Ipv4AddressHelper address ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodeInterface = address.Assign (staDevice);
  Ipv4InterfaceContainer apNodeInterface = address.Assign (apDevice);

  // PopulateArpCache
  PopulateARPcache ();

  // Configure applications
  DataRate applicationDataRate = DataRate (dataRate * 1e6);
  uint32_t portNumber = 9;

  for (uint32_t j = 0; j < wifiStaNodes.GetN (); ++j)
    {
      InstallTrafficGenerator (wifiStaNodes.Get (j), wifiApNode.Get (0), portNumber++,
                               applicationDataRate, packetSize, simulationTime, warmupTime, fuzzTime);
    }

  // Install FlowMonitor
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();
  Simulator::Schedule (Seconds (warmupTime), &GetWarmupFlows, monitor);

  // Generate PCAP at AP
  if (!pcapName.empty ())
    {
      phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
      phy.EnablePcap (pcapName, apDevice);
    }
  
  // Register callback for power change
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                   MakeCallback (PowerCallback));
  
  // Schedule all power changes
  double time;
  bool maxPower;

  // The interval between each change follows the exponential distribution
  Ptr<ExponentialRandomVariable> x = CreateObject<ExponentialRandomVariable> ();
  x->SetAttribute ("Mean", DoubleValue (interval));
  x->SetStream(-1);

  for (uint32_t j = 0; j < wifiStaNodes.GetN (); ++j)
  {
    time = warmupTime;
    maxPower = false;
    while (time < simulationTime)
    {
      time += x->GetValue ();
      Simulator::Schedule (Seconds (time), &ChangePower, wifiStaNodes.Get (j), maxPower);
      maxPower = !maxPower;
    }
  }

  // Register distance measurements
  for (uint32_t j = 0; j < wifiStaNodes.GetN (); ++j)
    {
      Simulator::ScheduleNow (&UpdateDistance, wifiStaNodes.Get (j), wifiApNode.Get (0));
    }

  // Define simulation stop time
  Simulator::Stop (Seconds (warmupTime + simulationTime));

  // Record start time
  std::cout << "Starting simulation..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now ();

  // Run the simulation!
  Simulator::Run ();

  // Record stop time and count duration
  auto finish = std::chrono::high_resolution_clock::now ();
  std::chrono::duration<double> elapsed = finish - start;

  std::cout << "Done!" << std::endl
            << "Elapsed time: " << elapsed.count () << " s" << std::endl
            << std::endl;

  // Calculate per-flow throughput and Jain's fairness index
  double nWifiReal = 0;
  double jainsIndexN = 0.;
  double jainsIndexD = 0.;

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
  std::cout << "Results: " << std::endl;

  for (auto &stat : stats)
    {
      double flow = (8 * stat.second.rxBytes - warmupFlows[stat.first]) / (1e6 * simulationTime);

      if (flow > 0)
        {
          nWifiReal += 1;
        }

      jainsIndexN += flow;
      jainsIndexD += flow * flow;

      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (stat.first);
      std::cout << "Flow " << stat.first << " (" << t.sourceAddress << " -> "
                << t.destinationAddress << ")\tThroughput: " << flow << " Mb/s" << std::endl;
    }

  double totalThr = jainsIndexN;
  double fairnessIndex = jainsIndexN * jainsIndexN / (nWifiReal * jainsIndexD);

  // Print results
  std::cout << std::endl
            << "Network throughput: " << totalThr << " Mb/s" << std::endl
            << "Jain's fairness index: " << fairnessIndex << std::endl
            << std::endl;

  // Gather results in CSV format
  double velocity = mobilityModel == "Distance" ? 0. : nodeSpeed;

  std::ostringstream csvOutput;
  csvOutput << mobilityModel << ',' << wifiManagerName << ',' << velocity << ',' << distance << ',' << nWifi << ','
            << nWifiReal << ',' << RngSeedManager::GetRun () << ',' << totalThr << std::endl;

  // Print results to std output
  std::cout << "mobility,manager,velocity,distance,nWifi,nWifiReal,seed,throughput"
            << std::endl
            << csvOutput.str ();

  // Print results to file
  std::ofstream outputFile (csvPath);
  outputFile << csvOutput.str ();
  std::cout << std::endl << "Simulation data saved to: " << csvPath << std::endl << std::endl;

  //Clean-up
  Simulator::Destroy ();

  return 0;
}

/***** Function definitions *****/

void
ChangePower (Ptr<Node> staNode, uint8_t powerLevel)
{
  // Override global variable with new power level
  globalPowerLevel = powerLevel;

  // Change power in STA
  std::stringstream devicePath;
  devicePath  << "/NodeList/" 
              << staNode->GetId () 
              << "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/DefaultTxPowerLevel";
  Config::Set (devicePath.str(), UintegerValue (powerLevel));
}

void
GetWarmupFlows (Ptr<FlowMonitor> monitor)
{
  for (auto &stat : monitor->GetFlowStats ())
    {
      warmupFlows.insert (std::pair<uint32_t, double> (stat.first, 8 * stat.second.rxBytes));
    }
}

void
InstallTrafficGenerator (Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, uint32_t port,
                         DataRate offeredLoad, uint32_t packetSize, double warmupTime,
                         double simulationTime, double fuzzTime)
{
  // Get sink address
  Ptr<Ipv4> ipv4 = toNode->GetObject<Ipv4> ();
  Ipv4Address addr = ipv4->GetAddress (1, 0).GetLocal ();

  // Define type of service
  uint8_t tosValue = 0x70; //AC_BE

  // Add random fuzz to app start time
  Ptr<UniformRandomVariable> fuzz = CreateObject<UniformRandomVariable> ();
  fuzz->SetAttribute ("Min", DoubleValue (0.));
  fuzz->SetAttribute ("Max", DoubleValue (fuzzTime));
  fuzz->SetStream(-1);
  double applicationsStart = fuzz->GetValue ();

  // Configure source and sink
  InetSocketAddress sinkSocket (addr, port);
  sinkSocket.SetTos (tosValue);
  PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", sinkSocket);

  OnOffHelper onOffHelper ("ns3::UdpSocketFactory", sinkSocket);
  onOffHelper.SetConstantRate (offeredLoad, packetSize);

  // Configure applications
  ApplicationContainer sinkApplications (packetSinkHelper.Install (toNode));
  ApplicationContainer sourceApplications (onOffHelper.Install (fromNode));

  sinkApplications.Start (Seconds (applicationsStart));
  sinkApplications.Stop (Seconds (warmupTime + simulationTime));
  sourceApplications.Start (Seconds (applicationsStart));
  sourceApplications.Stop (Seconds (warmupTime + simulationTime));
}

void
PopulateARPcache ()
{
  Ptr<ArpCache> arp = CreateObject<ArpCache> ();
  arp->SetAliveTimeout (Seconds (3600 * 24 * 365));

  for (auto i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
      ObjectVectorValue interfaces;
      ip->GetAttribute ("InterfaceList", interfaces);

      for (auto j = interfaces.Begin (); j != interfaces.End (); j++)
        {
          Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface> ();
          Ptr<NetDevice> device = ipIface->GetDevice ();
          Mac48Address addr = Mac48Address::ConvertFrom (device->GetAddress ());

          for (uint32_t k = 0; k < ipIface->GetNAddresses (); k++)
            {
              Ipv4Address ipAddr = ipIface->GetAddress (k).GetLocal ();
              if (ipAddr == Ipv4Address::GetLoopback ())
                {
                  continue;
                }

              ArpCache::Entry *entry = arp->Add (ipAddr);
              Ipv4Header ipv4Hdr;
              ipv4Hdr.SetDestination (ipAddr);

              Ptr<Packet> p = Create<Packet> (100);
              entry->MarkWaitReply (ArpCache::Ipv4PayloadHeaderPair (p, ipv4Hdr));
              entry->MarkAlive (addr);
            }
        }
    }

  for (auto i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Ipv4L3Protocol> ip = (*i)->GetObject<Ipv4L3Protocol> ();
      ObjectVectorValue interfaces;
      ip->GetAttribute ("InterfaceList", interfaces);

      for (auto j = interfaces.Begin (); j != interfaces.End (); j++)
        {
          Ptr<Ipv4Interface> ipIface = (*j).second->GetObject<Ipv4Interface> ();
          ipIface->SetAttribute ("ArpCache", PointerValue (arp));
        }
    }
}

void
PowerCallback (std::string path, Ptr<const Packet> packet, double txPowerW)
{
  size_t start = 10; // length of "/NodeList/" string
  size_t end = path.find ("/DeviceList/");
  std::string nodeId = path.substr (start, end - start);

  Config::Set ("/NodeList/" + nodeId +
                   "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                   "$ns3::MlWifiManager/Power",
               DoubleValue (10 * (3 + log10 (txPowerW))));

  Config::Set ("/NodeList/" + nodeId +
                   "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                   "$ns3::OracleWifiManager/Power",
               DoubleValue (10 * (3 + log10 (txPowerW))));
}

void
UpdateDistance (Ptr<Node> staNode, Ptr<Node> apNode)
{
  Ptr<MobilityModel> staMobility = staNode->GetObject<MobilityModel> ();
  Ptr<MobilityModel> apMobility = apNode->GetObject<MobilityModel> ();
  double d = staMobility->GetDistanceFrom (apMobility);

  Config::Set ("/NodeList/" + std::to_string (staNode->GetId ()) +
                   "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                   "$ns3::MlWifiManager/Distance",
               DoubleValue (d));

    Config::Set ("/NodeList/" + std::to_string (staNode->GetId ()) +
                   "/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/"
                   "$ns3::OracleWifiManager/Distance",
               DoubleValue (d));

  Simulator::Schedule (Seconds (DISTANCE_UPDATE_INTERVAL), &UpdateDistance, staNode, apNode);
}
