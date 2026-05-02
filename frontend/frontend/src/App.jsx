import React from 'react'
import UploadAndDisplay from './components/UploadAndDisplay'

export default function App(){
  return (
    <div className="app-bg">
      <div className="card glass">
        <h1 className="title">Offroad Segmentation Demo</h1>
        <UploadAndDisplay />
      </div>
    </div>
  )
}
