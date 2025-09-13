"use client"
import React, { useState } from 'react';
import { Upload, FileText, Search, Filter, MapPin, Clock, DollarSign, Briefcase, Star, TrendingUp, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import Header from '../dashboard/_components/Header';
import { UploadButton } from "@uploadthing/react";



const ResumePage = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [jobFilters, setJobFilters] = useState({
    location: '',
    experience: '',
    jobType: '',
    salary: '',
    company: ''
  });

  // Mock analysis results
  const mockAnalysisResults = {
    score: 85,
    strengths: [
      'Strong technical skills in React and Node.js',
      'Relevant work experience in software development',
      'Good educational background',
      'Clear project descriptions'
    ],
    improvements: [
      'Add more quantifiable achievements',
      'Include more industry-specific keywords',
      'Expand on leadership experience'
    ],
    keywords: ['React', 'Node.js', 'JavaScript', 'Python', 'AWS', 'MongoDB'],
    atsCompatibility: 'Good'
  };

  // Mock job listings
  const mockJobs = [
    {
      id: 1,
      title: 'Senior Frontend Developer',
      company: 'Tech Corp',
      location: 'San Francisco, CA',
      salary: '$120k - $160k',
      type: 'Full-time',
      experience: 'Senior',
      matchScore: 92,
      posted: '2 days ago'
    },
    {
      id: 2,
      title: 'React Developer',
      company: 'StartupXYZ',
      location: 'New York, NY',
      salary: '$90k - $120k',
      type: 'Full-time',
      experience: 'Mid-level',
      matchScore: 87,
      posted: '1 week ago'
    },
    {
      id: 3,
      title: 'Full Stack Engineer',
      company: 'Innovation Labs',
      location: 'Remote',
      salary: '$100k - $140k',
      type: 'Full-time',
      experience: 'Mid-level',
      matchScore: 78,
      posted: '3 days ago'
    }
  ];

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
    }
  };

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    // Simulate analysis delay
    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisResults(mockAnalysisResults);
    }, 3000);
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getMatchScoreColor = (score) => {
    if (score >= 85) return 'bg-green-500';
    if (score >= 70) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-black to-gray-900">
      {/* Header */}
      <Header />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Resume Upload Section */}
        <div className="bg-black bg-opacity-20 backdrop-blur-md rounded-2xl border border-purple-500/20 p-8 mb-8">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-400 to-pink-400 rounded-2xl flex items-center justify-center">
                <FileText className="h-8 w-8 text-white" />
              </div>
            </div>
            <h1 className="text-3xl font-bold text-white mb-4">
              Optimize Your Resume with AI
            </h1>
            <p className="text-purple-200 text-lg mb-8 max-w-2xl mx-auto">
              Upload your resume to get instant AI-powered feedback and find matching job opportunities
            </p>

            {!uploadedFile ? (
              <div className="border-2 border-dashed border-purple-400 rounded-xl p-8 hover:border-purple-300 transition-colors">
                <Upload className="h-12 w-12 text-purple-400 mx-auto mb-4" />
                <p className="text-white text-lg mb-4">Drop your resume here or click to browse</p>
                <p className="text-purple-300 text-sm mb-6">Supports PDF, DOC, DOCX (Max 5MB)</p>

                <UploadButton
                  endpoint="resumeUploader"
                  onClientUploadComplete={(res) => {
                    if (res && res.length > 0) {
                      setUploadedFile({ name: res[0].name, url: res[0].url });
                    }
                  }}
                  onUploadError={(error) => {
                    alert(`Upload failed: ${error.message}`);
                  }}
                  appearance={{
                    button:
                      "bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-xl font-semibold cursor-pointer hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105",
                    container: "flex justify-center",
                  }}
                />

              </div>
            ) : (
              <div className="bg-gray-800 bg-opacity-50 rounded-xl p-6">
                <div className="flex items-center justify-center space-x-4 mb-6">
                  <FileText className="h-8 w-8 text-green-400" />
                  <span className="text-white text-lg">{uploadedFile.name}</span>
                  <CheckCircle className="h-6 w-6 text-green-400" />
                </div>
                {!analysisResults && (
                  <button
                    onClick={() => handleAnalyze(uploadedFile.url)}
                    disabled={isAnalyzing}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-3 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isAnalyzing ? (
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        <span>Analyzing...</span>
                      </div>
                    ) : (
                      "Analyze Resume"
                    )}
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Analysis Results */}
        {analysisResults && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            {/* Overall Score */}
            <div className="bg-black bg-opacity-20 backdrop-blur-md rounded-2xl border border-purple-500/20 p-6">
              <div className="text-center">
                <TrendingUp className="h-8 w-8 text-purple-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Resume Score</h3>
                <div
                  className={`text-4xl font-bold mb-2 ${getScoreColor(
                    analysisResults.score
                  )}`}
                >
                  {analysisResults.score}/100
                </div>
                <p className="text-purple-300">
                  ATS Compatibility: {analysisResults.atsCompatibility}
                </p>
              </div>
            </div>

            {/* Strengths */}
            <div className="bg-black bg-opacity-20 backdrop-blur-md rounded-2xl border border-purple-500/20 p-6">
              <div className="flex items-center space-x-2 mb-4">
                <CheckCircle className="h-6 w-6 text-green-400" />
                <h3 className="text-xl font-semibold text-white">Strengths</h3>
              </div>
              <ul className="space-y-2">
                {analysisResults.strengths.map((strength, index) => (
                  <li
                    key={index}
                    className="text-purple-200 text-sm flex items-start space-x-2"
                  >
                    <div className="w-1.5 h-1.5 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>{strength}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Improvements */}
            <div className="bg-black bg-opacity-20 backdrop-blur-md rounded-2xl border border-purple-500/20 p-6">
              <div className="flex items-center space-x-2 mb-4">
                <AlertCircle className="h-6 w-6 text-yellow-400" />
                <h3 className="text-xl font-semibold text-white">Improvements</h3>
              </div>
              <ul className="space-y-2">
                {analysisResults.improvements.map((improvement, index) => (
                  <li
                    key={index}
                    className="text-purple-200 text-sm flex items-start space-x-2"
                  >
                    <div className="w-1.5 h-1.5 bg-yellow-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span>{improvement}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Job Search Filters */}
        <div className="bg-black bg-opacity-20 backdrop-blur-md rounded-2xl border border-purple-500/20 p-6 mb-8">
          <div className="flex items-center space-x-2 mb-6">
            <Search className="h-6 w-6 text-purple-400" />
            <h2 className="text-2xl font-bold text-white">Find Matching Jobs</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
            <div>
              <label className="block text-purple-300 text-sm mb-2">Location</label>
              <div className="relative">
                <MapPin className="absolute left-3 top-3 h-4 w-4 text-purple-400" />
                <input
                  type="text"
                  placeholder="San Francisco, CA"
                  className="w-full bg-gray-800 bg-opacity-50 border border-purple-500/30 rounded-lg pl-10 pr-4 py-2.5 text-white placeholder-gray-400 focus:outline-none focus:border-purple-400"
                  value={jobFilters.location}
                  onChange={(e) => setJobFilters({...jobFilters, location: e.target.value})}
                />
              </div>
            </div>

            <div>
              <label className="block text-purple-300 text-sm mb-2">Experience Level</label>
              <select
                className="w-full bg-gray-800 bg-opacity-50 border border-purple-500/30 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-purple-400"
                value={jobFilters.experience}
                onChange={(e) => setJobFilters({...jobFilters, experience: e.target.value})}
              >
                <option value="">All Levels</option>
                <option value="entry">Entry Level</option>
                <option value="mid">Mid Level</option>
                <option value="senior">Senior Level</option>
                <option value="executive">Executive</option>
              </select>
            </div>

            <div>
              <label className="block text-purple-300 text-sm mb-2">Job Type</label>
              <select
                className="w-full bg-gray-800 bg-opacity-50 border border-purple-500/30 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-purple-400"
                value={jobFilters.jobType}
                onChange={(e) => setJobFilters({...jobFilters, jobType: e.target.value})}
              >
                <option value="">All Types</option>
                <option value="full-time">Full-time</option>
                <option value="part-time">Part-time</option>
                <option value="contract">Contract</option>
                <option value="remote">Remote</option>
              </select>
            </div>

            <div>
              <label className="block text-purple-300 text-sm mb-2">Salary Range</label>
              <div className="relative">
                <DollarSign className="absolute left-3 top-3 h-4 w-4 text-purple-400" />
                <input
                  type="text"
                  placeholder="80k - 120k"
                  className="w-full bg-gray-800 bg-opacity-50 border border-purple-500/30 rounded-lg pl-10 pr-4 py-2.5 text-white placeholder-gray-400 focus:outline-none focus:border-purple-400"
                  value={jobFilters.salary}
                  onChange={(e) => setJobFilters({...jobFilters, salary: e.target.value})}
                />
              </div>
            </div>

            <div>
              <label className="block text-purple-300 text-sm mb-2">Company</label>
              <input
                type="text"
                placeholder="Company name"
                className="w-full bg-gray-800 bg-opacity-50 border border-purple-500/30 rounded-lg px-4 py-2.5 text-white placeholder-gray-400 focus:outline-none focus:border-purple-400"
                value={jobFilters.company}
                onChange={(e) => setJobFilters({...jobFilters, company: e.target.value})}
              />
            </div>
          </div>

          <button className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-2.5 rounded-lg font-semibold hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105 flex items-center space-x-2">
            <Search className="h-4 w-4" />
            <span>Search Jobs</span>
          </button>
        </div>

        {/* Job Listings */}
        <div className="bg-black bg-opacity-20 backdrop-blur-md rounded-2xl border border-purple-500/20 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">Recommended Jobs</h2>
            <span className="text-purple-300 text-sm">{mockJobs.length} jobs found</span>
          </div>

          <div className="space-y-4">
            {mockJobs.map((job) => (
              <div key={job.id} className="bg-gray-800 bg-opacity-30 rounded-xl p-6 border border-purple-500/10 hover:border-purple-500/30 transition-colors">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-white mb-2">{job.title}</h3>
                    <p className="text-purple-300 text-lg mb-2">{job.company}</p>
                    <div className="flex flex-wrap items-center gap-4 text-sm text-purple-200">
                      <div className="flex items-center space-x-1">
                        <MapPin className="h-4 w-4" />
                        <span>{job.location}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <DollarSign className="h-4 w-4" />
                        <span>{job.salary}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Briefcase className="h-4 w-4" />
                        <span>{job.type}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Clock className="h-4 w-4" />
                        <span>{job.posted}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="text-right">
                      <div className="text-white font-semibold text-sm">Match Score</div>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${getMatchScoreColor(job.matchScore)}`}></div>
                        <span className="text-white font-bold">{job.matchScore}%</span>
                      </div>
                    </div>
                    <Star className="h-5 w-5 text-yellow-400" />
                  </div>
                </div>
                <button className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-2 rounded-lg font-semibold hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105">
                  View Details
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResumePage;