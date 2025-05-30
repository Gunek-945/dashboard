import express from 'express';
import { DEFAULT_USER_PROFILE, SUGGESTED_ACTIONS_DATA, DOCUMENTS_DATA } from '../data/userData.js';

const router = express.Router();

// Get user profile data
router.get('/profile', (req, res) => {
  try {
    res.json({
      success: true,
      data: DEFAULT_USER_PROFILE,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to fetch user profile',
      message: error.message
    });
  }
});

// Get suggested actions with progressive loading
router.get('/actions', (req, res) => {
  try {
    const { 
      page = 1, 
      limit = 3, 
      status = null,
      loadMore = false 
    } = req.query;

    const pageNum = parseInt(page);
    const limitNum = parseInt(limit);
    
    // Filter actions by status if provided
    let filteredActions = SUGGESTED_ACTIONS_DATA;
    if (status) {
      filteredActions = SUGGESTED_ACTIONS_DATA.filter(action => 
        action.status.toLowerCase() === status.toLowerCase()
      );
    }

    // For initial load, show first 2-3 items
    // For subsequent loads, implement pagination
    const startIndex = loadMore === 'true' ? (pageNum - 1) * limitNum : 0;
    const endIndex = loadMore === 'true' ? startIndex + limitNum : Math.min(3, filteredActions.length);
    
    const paginatedActions = filteredActions.slice(startIndex, endIndex);
    
    const totalItems = filteredActions.length;
    const hasMore = endIndex < totalItems;
    const nextPage = hasMore ? pageNum + 1 : null;

    res.json({
      success: true,
      data: {
        actions: paginatedActions,
        pagination: {
          currentPage: pageNum,
          totalItems,
          itemsPerPage: limitNum,
          totalPages: Math.ceil(totalItems / limitNum),
          hasMore,
          nextPage,
          isInitialLoad: loadMore !== 'true'
        }
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to fetch suggested actions',
      message: error.message
    });
  }
});

// Get specific action by ID
router.get('/actions/:id', (req, res) => {
  try {
    const { id } = req.params;
    const action = SUGGESTED_ACTIONS_DATA.find(action => action.id === id);
    
    if (!action) {
      return res.status(404).json({
        success: false,
        error: 'Action not found',
        message: `No action found with ID: ${id}`
      });
    }

    res.json({
      success: true,
      data: action,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to fetch action details',
      message: error.message
    });
  }
});

// Get documents with progressive loading
router.get('/documents', (req, res) => {
  try {
    const { 
      page = 1, 
      limit = 3,
      type = null,
      loadMore = false 
    } = req.query;

    const pageNum = parseInt(page);
    const limitNum = parseInt(limit);
    
    // Filter documents by type if provided
    let filteredDocuments = DOCUMENTS_DATA;
    if (type) {
      filteredDocuments = DOCUMENTS_DATA.filter(doc => 
        doc.type.toLowerCase() === type.toLowerCase()
      );
    }

    // For initial load, show first 2-3 items
    // For subsequent loads, implement pagination
    const startIndex = loadMore === 'true' ? (pageNum - 1) * limitNum : 0;
    const endIndex = loadMore === 'true' ? startIndex + limitNum : Math.min(3, filteredDocuments.length);
    
    const paginatedDocuments = filteredDocuments.slice(startIndex, endIndex);
    
    const totalItems = filteredDocuments.length;
    const hasMore = endIndex < totalItems;
    const nextPage = hasMore ? pageNum + 1 : null;

    res.json({
      success: true,
      data: {
        documents: paginatedDocuments,
        pagination: {
          currentPage: pageNum,
          totalItems,
          itemsPerPage: limitNum,
          totalPages: Math.ceil(totalItems / limitNum),
          hasMore,
          nextPage,
          isInitialLoad: loadMore !== 'true'
        }
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to fetch documents',
      message: error.message
    });
  }
});

// Get dashboard summary (overview data)
router.get('/dashboard-summary', (req, res) => {
  try {
    const totalActions = SUGGESTED_ACTIONS_DATA.length;
    const completedActions = SUGGESTED_ACTIONS_DATA.filter(action => action.status === 'Completed').length;
    const inProgressActions = SUGGESTED_ACTIONS_DATA.filter(action => action.status === 'In Progress').length;
    const notStartedActions = SUGGESTED_ACTIONS_DATA.filter(action => action.status === 'Not Started').length;
    
    // Calculate onboarding completion percentage
    let completedValue = 0;
    SUGGESTED_ACTIONS_DATA.forEach(action => {
      if (action.status === 'Completed') {
        completedValue += 1;
      } else if (action.status === 'In Progress') {
        completedValue += (action.progressValue || 50) / 100;
      }
    });
    const onboardingCompletion = (completedValue / totalActions) * 100;

    res.json({
      success: true,
      data: {
        user: DEFAULT_USER_PROFILE,
        stats: {
          totalActions,
          completedActions,
          inProgressActions,
          notStartedActions,
          onboardingCompletion: Math.round(onboardingCompletion),
          totalDocuments: DOCUMENTS_DATA.length
        },
        recentActions: SUGGESTED_ACTIONS_DATA.slice(0, 3), // Show first 3 actions
        recentDocuments: DOCUMENTS_DATA.slice(0, 3) // Show first 3 documents
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to fetch dashboard summary',
      message: error.message
    });
  }
});

export default router; 